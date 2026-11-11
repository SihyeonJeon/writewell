"""Topic-matched comparison: pre-LLM Taiwanese human text vs raw model output on the same titles.

Reads data/human/*.jsonl and data/ai/<model>.jsonl. Writes out/zh-Hant-v2/{report.md,*.csv}.
Every candidate tell is listed with real example sentences so a native reader can judge it.
"""
import csv
import gc
import json
import math
import re
import resource
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ngram_zh import sentences, words  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "zh-Hant-v2"
OUT.mkdir(parents=True, exist_ok=True)

PARTICLES = list("啦喔耶吧呢啊囉嘛欸齁哦唷呀吼咧") + ["超", "蠻", "滿", "真的", "其實", "XD", "哈哈", "～", "⋯", "…"]
STRUCT = {
    "md_heading": r"(?m)^#{1,6} ",
    "md_bold": r"\*\*[^*\n]+\*\*",
    "bullet_line": r"(?m)^\s*[-*•] ",
    "numbered_line": r"(?m)^\s*\d+[.、)] ",
    "rule_line": r"(?m)^-{3,}\s*$",
    "label_colon_line": r"(?m)^[^\n，。]{2,12}：\s*$|^\*\*[^*\n]{2,14}[：:]\*\*",
    "em_dash": r"——|—",
    "placeholder": r"〔[^〕]*〕|\[[^\]]{2,20}\]|【[^】]*】",
    "emoji": r"[\U0001F300-\U0001FAFF☀-➿]",
    "quote_marks_tw": r"「",
    "quote_marks_cn": r"[“”]",
    "exclaim": r"[！!]",
    "question": r"[？?]",
    "tilde": r"[～~]",
    "ellipsis": r"…|⋯|\.{3}",
    "kaomoji_xd": r"XD|QQ|\^\^|>_<|orz|\(\S{2,8}\)",
    "paren": r"[（(]",
}


def load_dir(d):
    docs = []
    for f in sorted((ROOT / "data" / d).glob("*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            ss = sentences(r["text"])
            if len(ss) < 3:
                continue
            docs.append({"group": r.get("model") or "human", "source": r["source"], "genre": r["genre"], "url": r["url"],
                         "variant": r.get("variant", ""), "raw": r["text"], "sents": ss, "word": [words(s) for s in ss]})
    return docs


def per1k(docs, rx):
    n = sum(len(d["raw"]) for d in docs) / 1000
    return sum(len(re.findall(rx, d["raw"])) for d in docs) / n


def df(docs, rx):
    return 100 * sum(1 for d in docs if re.search(rx, d["raw"])) / len(docs)


def grams(docs, n, only=None):
    """(tf, df) with space-joined string keys. `only` restricts counting to a candidate set."""
    tf, dfc = Counter(), Counter()
    for d in docs:
        seen = set()
        for w in d["word"]:
            for i in range(len(w) - n + 1):
                g = " ".join(w[i:i + n])
                if only is None or g in only:
                    tf[g] += 1
                    seen.add(g)
        dfc.update(seen)
    return tf, dfc


def total_grams(docs, n):
    return sum(max(len(w) - n + 1, 0) for d in docs for w in d["word"])


def examples(docs, n, wanted, per=2):
    ex = defaultdict(list)
    for d in docs:
        for s, w in zip(d["sents"], d["word"]):
            for i in range(len(w) - n + 1):
                g = " ".join(w[i:i + n])
                if g in wanted and len(ex[g]) < per and s not in ex[g]:
                    ex[g].append(s)
    return ex


def guard(limit_gb=3.0):
    """Stop before this process can pressure the machine (macOS reports ru_maxrss in bytes)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    print(f"peak RSS {rss:.2f} GB", file=sys.stderr, flush=True)
    if rss > limit_gb:
        raise SystemExit(f"memory guard: {rss:.1f} GB > {limit_gb} GB")


MACHINE = ROOT / "data" / "machine"            # machine-written articles; local only, never committed
VERTICALS = [tuple(v) for v in json.loads((MACHINE / "verticals.json").read_text())] if (MACHINE / "verticals.json").exists() else []


def load_machine():
    docs = []
    for name in ("machine_a_zh-Hant", "machine_b_zh-Hant"):
        for r in json.load(open(MACHINE / f"{name}.json")):
            ss = sentences(r["text"])
            if len(ss) < 3:
                continue
            docs.append({"group": r["domain"].split(".")[0], "source": r["domain"], "genre": "machine", "url": r["url"],
                         "raw": r["text"], "sents": ss, "word": [words(s) for s in ss]})
    return docs


def main():
    H, A = load_dir("human"), load_machine()
    srcs = {d["source"] for d in H}
    models = sorted({d["group"] for d in A})
    by = {m: [d for d in A if d["group"] == m] for m in models}
    genres = sorted({d["genre"] for d in H})
    md = ["# zh-Hant: 사람 글(≤2021, 대만, 여러 장르) vs 기계 작성 글\n",
          f"사람 {len(H)}편({len(srcs)}개 출처) · " + " · ".join(f"{m} {len(by[m])}편" for m in models) + "\n"]

    # A. structure and punctuation
    md.append("\n## 1. 구조·문장부호 (문서 출현율 % / 1천 자당)\n\n| 항목 | 사람 | " + " | ".join(models) + " |\n|---|---|" + "---|" * len(models))
    for k, rx in STRUCT.items():
        row = [f"{df(H, rx):.0f}% / {per1k(H, rx):.2f}"] + [f"{df(by[m], rx):.0f}% / {per1k(by[m], rx):.2f}" for m in models]
        md.append(f"| {k} | " + " | ".join(row) + " |")

    # B. sentence shape
    md.append("\n## 2. 문장 길이 (글자)\n\n| 집단 | p10 | p50 | p90 | 표준편차 | 10자 미만 % | 60자 초과 % |\n|---|---|---|---|---|---|---|")
    for name, docs in [("사람", H)] + [(m, by[m]) for m in models]:
        L = np.array([len(s) for d in docs for s in d["sents"]])
        md.append(f"| {name} | {np.percentile(L, 10):.0f} | {np.median(L):.0f} | {np.percentile(L, 90):.0f} | {L.std():.1f} | "
                  f"{100 * (L < 10).mean():.1f} | {100 * (L > 60).mean():.1f} |")

    # C. particles and colloquial markers, by genre
    md.append("\n## 3. 어기사·구어 표지 (1천 자당)\n\n| 낱말 | 사람 | " + " | ".join(models) + " |\n|---|---|" + "---|" * len(models))
    for p in PARTICLES:
        rx = re.escape(p)
        h = per1k(H, rx)
        if h < 0.05 and all(per1k(by[m], rx) < 0.05 for m in models):
            continue
        md.append(f"| {p} | {h:.2f} | " + " | ".join(f"{per1k(by[m], rx):.2f}" for m in models) + " |")

    # D. mainland vocabulary (OpenCC TWPhrases: mainland term -> Taiwan term)
    pairs = {}
    for line in open(ROOT / "data" / "lex" / "TWPhrases.txt", encoding="utf-8"):
        a, _, b = line.strip().partition("\t")
        if len(a) >= 2 and b and a not in b.split(" "):
            pairs[a] = b
    ht, at = "".join(d["raw"] for d in H), "".join(d["raw"] for d in A)
    rows = []
    for a, b in pairs.items():
        ca, ch = at.count(a), ht.count(a)
        if ca + ch >= 3:
            rows.append((a, b, ca, ch, 1e6 * ca / len(at), 1e6 * ch / len(ht)))
    rows.sort(key=lambda r: -(r[4] - r[5]))
    md.append(f"\n## 4. 대륙 용어 (OpenCC 대응표 {len(pairs)}쌍, 100만 자당)\n\n"
              f"전체 합계: 사람 {sum(r[5] for r in rows):.0f} · AI {sum(r[4] for r in rows):.0f}\n\n| 대륙 | 대만 | AI | 사람 |\n|---|---|---|---|")
    for a, b, _ca, _ch, ra, rh in rows[:25]:
        md.append(f"| {a} | {b} | {ra:.0f} | {rh:.0f} |")

    # E. lexical tells. Memory-lean: string keys, human counts only for candidate grams,
    #    example sentences collected in a second pass for the rows that are printed.
    for n, top in ((1, 60), (2, 60), (3, 40)):
        tfa, dfa = grams(A, n)
        na = sum(tfa.values())
        pm = {m: grams(by[m], n)[1] for m in models}
        cand, spread = set(), {}
        for g, c in tfa.items():
            if c < 8:
                continue
            share = {m: 100 * pm[m].get(g, 0) / len(by[m]) for m in models}
            if all(max(share[m] for m in v) >= 8 for v in VERTICALS):   # in every vertical, else a topic word
                cand.add(g)
                vs = [max(share[m] for m in v) for v in VERTICALS]
                spread[g] = round(min(vs) / max(vs), 2)      # 1.0 = same document rate in every vertical
        del pm
        keep_all = n <= 2                      # full human table only where it is small enough
        tfh, dfh = grams(H, n, None if keep_all else cand)
        nh = total_grams(H, n)
        gen_h, gen_n = {}, {}
        for ge in genres:
            sub = [d for d in H if d["genre"] == ge]
            gen_h[ge], gen_n[ge] = grams(sub, n, cand)[0], max(total_grams(sub, n), 1)
        guard()
        scale = 0.01 * min(na, nh) / (na + nh)

        def z(g):
            ya, yh = tfa.get(g, 0), tfh.get(g, 0)
            al = (ya + yh) * scale + 0.01
            d = math.log((ya + al) / (na - ya)) - math.log((yh + al) / (nh - yh))
            return d / math.sqrt(1 / (ya + al) + 1 / (yh + al))

        rows = []
        for g in cand:
            wins = sum(1 for ge in genres if tfa[g] / na > 2 * (gen_h[ge].get(g, 0) + 0.5) / gen_n[ge])
            rows.append([g, round(z(g), 1), round(1000 * tfa[g] / na, 2), round(1000 * tfh.get(g, 0) / nh, 2),
                         round(100 * dfa[g] / len(A), 1), round(100 * dfh.get(g, 0) / len(H), 1), f"{wins}/{len(genres)}", spread[g]])
        rows.sort(key=lambda r: -r[1])
        rows = rows[:600]
        ex = examples(A, n, {r[0] for r in rows[:top]})
        with open(OUT / f"tells_word{n}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gram", "z", "per1k_machine", "per1k_h", "df_machine", "df_h", "genres_over_2x", "client_spread", "examples"])
            w.writerows(r + [" ‖ ".join(ex.get(r[0], []))] for r in rows)
        md.append(f"\n## 5.{n} 기계 쪽으로 기우는 {n}-낱말 조합 (상위 {top}, 세 업종 모두에 나오는 것만)\n\n"
                  "문서 출현율 기계 / 사람 · 사람 장르 중 기계가 2배 넘게 쓰는 장르 수\n")
        for r in rows[:top]:
            md.append(f"- **{r[0]}** z={r[1]} · {r[4]}% / {r[5]}% · {r[6]}\n  - {' ‖ '.join(ex.get(r[0], []))[:170]}")
        if keep_all:
            rows_h = sorted(((g, z(g)) for g, c in tfh.items() if c >= 30), key=lambda r: r[1])[:40]
            md.append(f"\n### 사람 쪽으로 기우는 {n}-낱말 조합 (기계 % / 사람 %)\n\n" + " · ".join(
                f"{g} ({100 * dfa.get(g, 0) / len(A):.0f} / {100 * dfh.get(g, 0) / len(H):.0f})" for g, _ in rows_h))
        del tfa, dfa, tfh, dfh, gen_h, cand
        gc.collect()

    (OUT / "report.md").write_text("\n".join(md))
    print("\n".join(md[:70]))


if __name__ == "__main__":
    main()
