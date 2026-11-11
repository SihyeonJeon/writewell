"""Build the zh-Hant style gate from measurements. No hand-picked words.

Inputs
  out/zh-Hant-v2/tells_word{1,2,3}.csv   machine-written articles vs pre-LLM human text (calc/compare.py)
  data/human/*.jsonl                     pre-LLM Taiwanese text, 10 genres
  data/ai/*.jsonl                        raw model output on general topics (second evidence source)

Decision rules (all from data)
  candidate   over-used in every vertical of the machine corpus AND >2x the human rate in every human genre
              AND document rate >= 3x human
  uniformity  the machine corpus spans several clients; min/max rate across its verticals must be >= 0.3
  topic guard a noun-like word stays only if raw model output on unrelated topics also over-uses it
              (otherwise it is a subject word of the machine corpus, not a habit)
  ban         human document rate < 8% and machine/human ratio >= 5
  cap         everything else: allowed up to 1.5x the human rate for the text length (at least 1)
  shape       sentence-length spread floor and question-mark ceiling from human article genres

Output: gate/zh-Hant.json (words and numbers only; no source sentences)
"""
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import jieba.posseg as pseg
import numpy as np
from opencc import OpenCC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "calc"))
from ngram_zh import sentences, words  # noqa: E402

T2S = OpenCC("t2s")
ARTICLE_GENRES = {"magazine", "feature", "lifestyle", "health", "tech", "essay", "personal_blog"}
PUNCT = set("，。！？；：、「」『』（）《》…—")


def load(d):
    out = []
    for f in sorted((ROOT / "data" / d).glob("*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            ss = sentences(r["text"])
            if len(ss) >= 3:
                out.append({"genre": r["genre"], "model": r.get("model"), "raw": r["text"], "sents": ss, "word": [words(s) for s in ss]})
    return out


def rates(docs, wanted):
    """per-1k-token rate for each wanted gram (space-joined)."""
    ns = {len(g.split(" ")) for g in wanted}
    tf, total = Counter(), Counter()
    for d in docs:
        for w in d["word"]:
            for n in ns:
                total[n] += max(len(w) - n + 1, 0)
                for i in range(len(w) - n + 1):
                    g = " ".join(w[i:i + n])
                    if g in wanted:
                        tf[g] += 1
    return {g: (1000 * tf[g] / max(total[len(g.split(" "))], 1), tf[g]) for g in wanted}


def noun_like(tok):
    p = next(iter(pseg.cut(T2S.convert(tok), HMM=False))).flag
    return p[0] in "nst" or p in ("vn", "an")


def main():
    cand = {}
    for n in (1, 2, 3):
        rows = list(csv.DictReader(open(ROOT / "out" / "zh-Hant-v2" / f"tells_word{n}.csv")))
        for r in rows:
            a, b = r["genres_over_2x"].split("/")
            toks = r["gram"].split(" ")
            if a != b or any(t in ("９", "Ａ") for t in toks) or all(t in PUNCT for t in toks):
                continue
            dfa, dfh = float(r["df_machine"]), float(r["df_h"])
            if dfa / max(dfh, 0.1) < 3:
                continue
            cand[r["gram"]] = {"n": n, "df_machine": dfa, "df_human": dfh, "ratio": round(dfa / max(dfh, 0.1), 1),
                               "h_per1k": float(r["per1k_h"]), "spread": float(r["client_spread"])}
    print("candidates", len(cand), file=sys.stderr)

    H, R = load("human"), load("ai")
    Hm = [d for d in H if d["genre"] in {r["genre"] for r in R}]     # raw model output is compared with the same genres
    hr, rr = rates(Hm, set(cand)), rates(R, set(cand))

    # topic guard on noun-like tokens
    tokens = {t for g in cand for t in g.split(" ") if t not in PUNCT}
    tok_h, tok_r = rates(Hm, tokens), rates(R, tokens)
    topical = {t for t in tokens if noun_like(t) and not (tok_r[t][1] >= 3 and tok_r[t][0] >= 1.5 * tok_h[t][0])}
    # uniformity guard: a habit shows up at a similar rate for every client; a subject word does not
    uneven = {g for g, c in cand.items() if c["spread"] < 0.3}
    kept = {g: c for g, c in cand.items() if g not in uneven and not any(t in topical for t in g.split(" "))}
    print("dropped as uneven across clients:", " | ".join(sorted(uneven)), file=sys.stderr)
    print("dropped as topic:", len(cand) - len(kept), "| topical tokens:", " ".join(sorted(topical)), file=sys.stderr)

    ban, cap = {}, {}
    for g, c in kept.items():
        c["raw_model_per1k"] = round(rr[g][0], 3)
        if c["df_human"] < 8 and c["ratio"] >= 5:
            ban[g] = c
        else:
            cap[g] = c | {"allow_per1k": round(1.5 * c["h_per1k"], 3)}
    # an n-gram that contains a banned shorter gram is redundant
    def covered(g):
        t = g.split(" ")
        return any(" ".join(t[i:j]) in ban for i in range(len(t)) for j in range(i + 1, len(t) + 1) if (i, j) != (0, len(t)))
    ban = {g: c for g, c in ban.items() if not covered(g)}
    cap = {g: c for g, c in cap.items() if not covered(g)}

    # second source: raw model output on the same titles as human posts (topic is matched by construction)
    def doc_freq(docs):
        c = Counter()
        for d in docs:
            seen = set()
            for w in d["word"]:
                for n in (1, 2, 3):
                    for i in range(len(w) - n + 1):
                        seen.add(" ".join(w[i:i + n]))
            c.update(seen)
        return c
    dr, dh = doc_freq(R), doc_freq(Hm)
    by_model = [doc_freq([d for d in R if d["model"] == m]) for m in sorted({d["model"] for d in R})]
    sizes = [sum(1 for d in R if d["model"] == m) for m in sorted({d["model"] for d in R})]
    raw_ban = {}
    for g, c in dr.items():
        toks = g.split(" ")
        if any(t in ("９", "Ａ") for t in toks) or all(t in PUNCT for t in toks):
            continue
        a, h = 100 * c / len(R), 100 * dh.get(g, 0) / len(Hm)
        if a >= 10 and h < 2 and a / max(h, 0.1) >= 8 and all(100 * bm.get(g, 0) / n >= 6 for bm, n in zip(by_model, sizes)):
            raw_ban[g] = {"n": len(toks), "df_machine": round(a, 1), "df_human": round(h, 1), "ratio": round(a / max(h, 0.1), 1),
                          "source": "raw_model"}
    raw_ban = {g: c for g, c in raw_ban.items()
               if not any(" ".join(g.split(" ")[i:j]) in raw_ban or " ".join(g.split(" ")[i:j]) in ban
                          for i in range(c["n"]) for j in range(i + 1, c["n"] + 1) if (i, j) != (0, c["n"]))}
    print("RAW-MODEL BAN:", " | ".join(f"{g}({c['df_machine']}/{c['df_human']})" for g, c in sorted(raw_ban.items(), key=lambda kv: -kv[1]["ratio"])))
    ban = raw_ban | ban

    # shape thresholds from human article-like documents
    art = [d for d in H if d["genre"] in ARTICLE_GENRES and len(d["sents"]) >= 8]
    sd = np.array([np.std([len(s) for s in d["sents"]]) for d in art])
    q = np.array([1000 * len(re.findall(r"[？?]", d["raw"])) / len(d["raw"]) for d in art])
    long_share = np.array([np.mean([len(s) > 60 for s in d["sents"]]) for d in art])
    shape = {"sentence_len_sd_min": round(float(np.percentile(sd, 10)), 1),
             "question_per1k_chars_max": round(float(np.percentile(q, 90)), 2),
             "long_sentence_share_min": round(float(np.percentile(long_share, 25)), 3),
             "basis_docs": len(art)}

    out = {"lang": "zh-Hant",
           "basis": {"human_docs": len(H), "raw_model_docs": len(R), "rule": __doc__.split("Decision rules")[1].split("Output")[0].strip()},
           "ban": dict(sorted(ban.items(), key=lambda kv: -kv[1]["ratio"])),
           "cap": dict(sorted(cap.items(), key=lambda kv: -kv[1]["ratio"])),
           "shape": shape}
    (ROOT / "gate" / "zh-Hant.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"ban {len(ban)} · cap {len(cap)} · shape {shape}")
    print("BAN:", " | ".join(ban))
    print("CAP:", " | ".join(f"{g}({c['allow_per1k']})" for g, c in cap.items()))


if __name__ == "__main__":
    main()
