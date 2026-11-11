"""zh-Hant: AI-written vs pre-LLM human text. n-gram distributions and a two-LM likelihood ratio.

Corpora (not in git) are read from $WW_CORPUS:
  machine_a_zh-Hant.json (A1) · machine_b_zh-Hant.json (A2) · human_prellm_zh-Hant.json (H)

Outputs to out/zh-Hant/. No LLM calls.
"""
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jieba
import numpy as np
from opencc import OpenCC

CORPUS = Path(os.environ.get("WW_CORPUS", Path(__file__).resolve().parent.parent / "data" / "machine"))
OUT = Path(__file__).resolve().parent.parent / "out" / "zh-Hant"
OUT.mkdir(parents=True, exist_ok=True)
random.seed(7)
jieba.setLogLevel(60)
T2S = OpenCC("t2s")

LAT, NUM = "Ａ", "９"          # placeholders: latin run, number run
PUNCT = "，。！？；：、「」『』（）《》…—"
KEEP = re.compile(rf"[^一-鿿㐀-䶿{LAT}{NUM}{PUNCT}\n]")
HALF2FULL = str.maketrans({",": "，", "!": "！", "?": "？", ";": "；", ":": "：", "(": "（", ")": "）"})
HELD_OUT_H = ("skin168", "ton-yen", "health.udn", "drhhwang", "dr-skin")   # test-only human sources


def normalize(text: str) -> str:
    t = re.sub(r"https?://\S+", " ", text)
    t = re.sub(r"\d[\d,.\-~％%/:]*", NUM, t)
    t = re.sub(r"[A-Za-z][A-Za-z'’.&+\- ]*", LAT, t)
    t = t.translate(HALF2FULL)
    return KEEP.sub("", t)


def sentences(text: str):
    """Sentence strings (punctuation kept). Lines are hard boundaries."""
    out = []
    for line in normalize(text).split("\n"):
        for s in re.findall(r"[^。！？；]+[。！？；]?", line):
            if len(re.findall(r"[一-鿿]", s)) >= 4:
                out.append(s)
    return out


def words(sent: str):
    s = T2S.convert(sent)
    if len(s) != len(sent):          # conversion changed length: segment the original
        return [w for w in jieba.cut(sent, HMM=False)]
    toks, i = [], 0
    for w in jieba.cut(s, HMM=False):
        toks.append(sent[i:i + len(w)])
        i += len(w)
    return toks


def load(name):
    docs = json.load(open(CORPUS / f"{name}.json"))
    out = []
    for d in docs:
        ss = sentences(d["text"])
        if ss:
            out.append({"src": d.get("domain") or d.get("source"), "title": d.get("title", ""),
                        "char": [list(s) for s in ss], "word": [words(s) for s in ss]})
    return out


# ---------- distribution statistics ----------
def ngram_counts(docs, unit, n):
    tf, df = Counter(), Counter()
    for d in docs:
        seen = set()
        for s in d[unit]:
            for i in range(len(s) - n + 1):
                g = tuple(s[i:i + n])
                tf[g] += 1
                seen.add(g)
        df.update(seen)
    return tf, df


def shape(tf):
    N = sum(tf.values())
    p = np.array(sorted(tf.values(), reverse=True), dtype=float) / N
    H = float(-(p * np.log2(p)).sum())
    cum = np.cumsum(p)
    return {"tokens": N, "types": len(tf), "entropy_bits": round(H, 3),
            "top100_mass": round(float(cum[min(99, len(cum) - 1)]), 4),
            "types_for_half_mass": int(np.searchsorted(cum, 0.5) + 1),
            "hapax_share": round(sum(1 for v in tf.values() if v == 1) / len(tf), 4)}


def log_odds(tfa, tfh):
    """Monroe et al. weighted log-odds with an informative Dirichlet prior (z-score)."""
    prior = tfa + tfh
    a0 = sum(prior.values())
    na, nh = sum(tfa.values()), sum(tfh.values())
    scale = 0.01 * min(na, nh) / a0          # prior worth 1% of the smaller corpus
    z = {}
    for g, c in prior.items():
        al = c * scale
        ya, yh = tfa.get(g, 0), tfh.get(g, 0)
        d = math.log((ya + al) / (na + a0 * scale - ya - al)) - math.log((yh + al) / (nh + a0 * scale - yh - al))
        z[g] = d / math.sqrt(1 / (ya + al) + 1 / (yh + al))
    return z


# ---------- Witten-Bell interpolated n-gram LM ----------
class LM:
    def __init__(self, order):
        self.n = order
        self.c = [Counter() for _ in range(order + 1)]      # c[k][gram of length k]
        self.follow = [defaultdict(set) for _ in range(order)]  # follow[k][history of length k] -> set

    def fit(self, sents):
        for s in sents:
            seq = ["<s>"] * (self.n - 1) + list(s) + ["</s>"]
            for i in range(self.n - 1, len(seq)):
                for k in range(1, self.n + 1):
                    g = tuple(seq[i - k + 1:i + 1])
                    self.c[k][g] += 1
                    self.follow[k - 1][g[:-1]].add(g[-1])
        self.N = sum(self.c[1].values())
        self.hist = [Counter() for _ in range(self.n)]
        for k in range(1, self.n + 1):
            for g, v in self.c[k].items():
                self.hist[k - 1][g[:-1]] += v
        self.T = [{h: len(f) for h, f in fl.items()} for fl in self.follow]
        return self

    def logp(self, w, hist, V):
        """log2 P(w | hist), backing off through shorter histories."""
        t0 = self.T[0][()]
        p = (self.c[1].get((w,), 0) + t0 / V) / (self.N + t0)
        for k in range(1, self.n):
            h = tuple(hist[-k:])
            ch = self.hist[k].get(h, 0)
            if not ch:
                break
            t = self.T[k][h]
            p = (self.c[k + 1].get(h + (w,), 0) + t * p) / (ch + t)
        return math.log2(p)

    def score(self, s, V):
        seq = ["<s>"] * (self.n - 1) + list(s) + ["</s>"]
        return [self.logp(seq[i], seq[i - self.n + 1:i], V) for i in range(self.n - 1, len(seq))]


def auc(pos, neg):
    x = np.concatenate([pos, neg])
    r = x.argsort().argsort().astype(float) + 1
    # average ranks for ties
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, r)
    r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def balanced(docs, unit, budget):
    """Shuffle docs and take sentences until the token budget is reached."""
    docs = docs[:]
    random.shuffle(docs)
    out, n = [], 0
    for d in docs:
        for s in d[unit]:
            out.append(s)
            n += len(s)
        if n >= budget:
            break
    return out


def evaluate(train_a, train_h, test_a, test_h, unit, order):
    budget = min(sum(len(s) for d in train_a for s in d[unit]), sum(len(s) for d in train_h for s in d[unit]))
    sa, sh = balanced(train_a, unit, budget), balanced(train_h, unit, budget)
    la, lh = LM(order).fit(sa), LM(order).fit(sh)
    V = len(set(la.c[1]) | set(lh.c[1])) + 1

    def llr(s):
        a, h = la.score(s, V), lh.score(s, V)
        return [x - y for x, y in zip(a, h)]

    def run(docs):
        dsc, ssc = [], []
        for d in docs:
            per = [llr(s) for s in d[unit]]
            flat = [x for p in per for x in p]
            dsc.append(sum(flat) / len(flat))
            ssc += [sum(p) / len(p) for p, s in zip(per, d[unit]) if len(s) >= (8 if unit == "char" else 5)]
        return np.array(dsc), np.array(ssc)

    da, sa_ = run(test_a)
    dh, sh_ = run(test_h)
    cut = {q: float(np.percentile(sh_, q)) for q in (90, 95, 99)}
    return {"unit": unit, "order": order, "train_tokens_each": budget,
            "doc_auc": round(auc(da, dh), 4), "sent_auc": round(auc(sa_, sh_), 4),
            "n_docs": [len(da), len(dh)], "n_sents": [len(sa_), len(sh_)],
            "sent_llr_bits_per_token": {"ai_p10_50_90": [round(float(np.percentile(sa_, q)), 3) for q in (10, 50, 90)],
                                        "human_p10_50_90": [round(float(np.percentile(sh_, q)), 3) for q in (10, 50, 90)]},
            "ai_sents_flagged_at_human_pct": {f"p{q}": round(float((sa_ > c).mean()), 4) for q, c in cut.items()},
            "doc_score_by_source": by_source(test_a, da) | by_source(test_h, dh)}, (la, lh, V)


def by_source(docs, sc):
    g = defaultdict(list)
    for d, x in zip(docs, sc):
        g[d["src"]].append(x)
    return {k: [len(v), round(float(np.median(v)), 3)] for k, v in g.items()}


def main():
    A1, A2, H = load("machine_a_zh-Hant"), load("machine_b_zh-Hant"), load("human_prellm_zh-Hant")
    A = A1 + A2
    # style-only view: keep words common to all three corpora (DF >= 8% each), mask the rest as one slot
    dfs = [ngram_counts(x, "word", 1)[1] for x in (A1, A2, H)]
    common = {g[0] for g in dfs[2] if all(df.get(g, 0) / len(x) >= 0.08 for df, x in zip(dfs, (A1, A2, H)))}
    for d in A + H:
        d["wordm"] = []
        for s in d["word"]:
            m = []
            for w in s:
                t = w if w in common else "□"
                if not (t == "□" and m and m[-1] == "□"):
                    m.append(t)
            d["wordm"].append(m)
    json.dump(sorted(common), open(OUT / "common_vocab.json", "w"), ensure_ascii=False)
    summary = {"common_vocab": len(common), "docs": {"A1": len(A1), "A2": len(A2), "H": len(H)}, "shape": {}, "sentence": {}, "eval": []}

    # sentence construction
    for name, docs in (("A1", A1), ("A2", A2), ("H", H)):
        cl = np.array([len(s) for d in docs for s in d["char"]])
        wl = np.array([len(s) for d in docs for s in d["word"]])
        allc = "".join("".join(s) for d in docs for s in d["char"])
        k = len(allc) / 1000
        summary["sentence"][name] = {
            "sentences": len(cl), "chars": len(allc),
            "len_chars_p10_50_90": [int(np.percentile(cl, q)) for q in (10, 50, 90)],
            "len_words_p50": int(np.median(wl)), "len_chars_sd": round(float(cl.std()), 1),
            "per_1k_chars": {p: round(allc.count(p) / k, 2) for p in "，、：？！；「（—"}}

    # distribution shape + tells
    for unit, ns in (("char", (1, 2, 3, 4)), ("word", (1, 2, 3)), ("wordm", (2, 3, 4))):
        for n in ns:
            tfa1, dfa1 = ngram_counts(A1, unit, n)
            tfa2, dfa2 = ngram_counts(A2, unit, n)
            tfh, dfh = ngram_counts(H, unit, n)
            tfa = tfa1 + tfa2
            summary["shape"][f"{unit}{n}"] = {"A": shape(tfa), "H": shape(tfh)}
            z = log_odds(tfa, tfh)
            na, nh = sum(tfa.values()), sum(tfh.values())
            rows = []
            for g, zz in z.items():
                if tfa.get(g, 0) + tfh.get(g, 0) < 20:
                    continue
                rows.append(["".join(g) if unit == "char" else " ".join(g), round(zz, 2),
                             tfa.get(g, 0), tfh.get(g, 0),
                             round(1000 * tfa.get(g, 0) / na, 3), round(1000 * tfh.get(g, 0) / nh, 3),
                             round(100 * dfa1.get(g, 0) / len(A1), 1), round(100 * dfa2.get(g, 0) / len(A2), 1),
                             round(100 * dfh.get(g, 0) / len(H), 1)])
            rows.sort(key=lambda r: -r[1])
            with open(OUT / f"tells_{unit}{n}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["gram", "z", "tf_ai", "tf_h", "per1k_ai", "per1k_h", "df_a1", "df_a2", "df_h"])
                w.writerows(rows[:400] + rows[-400:])
            # mass of AI text sitting on grams that are >=4x over human rate
            over = sum(c for g, c in tfa.items() if c / na >= 4 * (tfh.get(g, 0) + 0.5) / nh and c >= 20)
            summary["shape"][f"{unit}{n}"]["ai_mass_on_4x_grams"] = round(over / na, 4)

    # next-word conditional probability: P(w2|w1) in AI vs human
    bia, _ = ngram_counts(A, "wordm", 2)
    bih, _ = ngram_counts(H, "wordm", 2)
    ua, uh = Counter(), Counter()
    for (a, _b), c in bia.items():
        ua[a] += c
    for (a, _b), c in bih.items():
        uh[a] += c
    rows = []
    for (a, b), c in bia.items():
        if c < 40 or uh[a] < 40:
            continue
        pa, ph = c / ua[a], (bih.get((a, b), 0) + 0.5) / (uh[a] + 1)
        rows.append([a, b, c, bih.get((a, b), 0), round(pa, 4), round(ph, 4), round(math.log2(pa / ph), 2)])
    rows.sort(key=lambda r: -r[6])
    with open(OUT / "transitions_word.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["w1", "w2", "n_ai", "n_h", "p_ai(w2|w1)", "p_h(w2|w1)", "log2_ratio"])
        w.writerows(rows[:500])

    # two-LM likelihood ratio, tested across client and across human source
    h_test = [d for d in H if any(k in d["src"] for k in HELD_OUT_H)]
    h_train = [d for d in H if d not in h_test]
    for unit, order in (("char", 2), ("char", 3), ("char", 4), ("word", 2), ("word", 3), ("wordm", 2), ("wordm", 3), ("wordm", 4)):
        for tag, tr, te in (("train set A -> test set B", A1, A2), ("train set B -> test set A", A2, A1)):
            r, _ = evaluate(tr, h_train, te, h_test, unit, order)
            r["split"] = tag
            summary["eval"].append(r)
            print(tag, unit, order, "doc AUC", r["doc_auc"], "sent AUC", r["sent_auc"],
                  "flag@H95", r["ai_sents_flagged_at_human_pct"]["p95"], file=sys.stderr)

    json.dump(summary, open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in ("docs", "sentence")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
