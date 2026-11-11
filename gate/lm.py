"""Word-combination probability: two trigram models (machine text, human text) and their likelihood ratio.

A sentence scores high when its word sequences are more probable under machine text than under human text.
Subject words are masked to one slot, so the score reads sentence construction, not topic.

Build:  python -m gate.lm build     (needs local corpora; writes models/zh-Hant-lm.pkl, not committed)
Use:    from gate.lm import sentence_scores
"""
import json
import math
import pickle
import random
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .check import split, tokens

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "zh-Hant-lm.pkl"
ORDER = 3
SLOT = "□"


def mask(toks, vocab):
    out = []
    for w in toks:
        t = w if w in vocab else SLOT
        if not (t == SLOT and out and out[-1] == SLOT):
            out.append(t)
    return out


def train(sents):
    """Witten-Bell interpolated trigram counts."""
    c = [Counter() for _ in range(ORDER + 1)]
    follow = [dict() for _ in range(ORDER)]
    for s in sents:
        seq = ["<s>"] * (ORDER - 1) + s + ["</s>"]
        for i in range(ORDER - 1, len(seq)):
            for k in range(1, ORDER + 1):
                g = tuple(seq[i - k + 1:i + 1])
                c[k][g] += 1
                follow[k - 1].setdefault(g[:-1], set()).add(g[-1])
    hist = [Counter() for _ in range(ORDER)]
    for k in range(1, ORDER + 1):
        for g, v in c[k].items():
            hist[k - 1][g[:-1]] += v
    return {"c": [dict(x) for x in c], "hist": [dict(x) for x in hist],
            "T": [{h: len(f) for h, f in fl.items()} for fl in follow], "N": sum(c[1].values())}


def logp(m, w, h, V):
    t0 = m["T"][0][()]
    p = (m["c"][1].get((w,), 0) + t0 / V) / (m["N"] + t0)
    for k in range(1, ORDER):
        hh = tuple(h[-k:])
        ch = m["hist"][k].get(hh, 0)
        if not ch:
            break
        t = m["T"][k][hh]
        p = (m["c"][k + 1].get(hh + (w,), 0) + t * p) / (ch + t)
    return math.log2(p)


def llr(model, toks):
    s = mask(toks, model["vocab"])
    if len(s) < 5:
        return None
    seq = ["<s>"] * (ORDER - 1) + s + ["</s>"]
    d = [logp(model["machine"], seq[i], seq[i - ORDER + 1:i], model["V"]) - logp(model["human"], seq[i], seq[i - ORDER + 1:i], model["V"])
         for i in range(ORDER - 1, len(seq))]
    return sum(d) / len(d)


@lru_cache
def load():
    return pickle.loads(MODEL.read_bytes()) if MODEL.exists() else None


def sentence_scores(text):
    """[(sentence index, score, flagged)] ; empty when no model is installed."""
    m = load()
    if not m:
        return []
    out = []
    for i, s in enumerate(split(text)):
        v = llr(m, tokens(s))
        if v is not None:
            out.append((i, round(v, 3), v > m["threshold"]))
    return out


def _docs(dirname, key=None):
    out = []
    for f in sorted((ROOT / "data" / dirname).glob("*.json*")):
        if "human_prellm" in f.name or f.name == "verticals.json":
            continue
        rows = json.load(open(f)) if f.suffix == ".json" else [json.loads(x) for x in open(f)]
        for r in rows:
            ss = [tokens(s) for s in split(r["text"])]
            ss = [s for s in ss if len(s) >= 5]
            if len(ss) >= 3:
                out.append({"group": f.stem if key is None else r.get(key), "sents": ss})
    return out


def _auc(pos, neg):
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + j + 1) / 2
        rank_sum += r * sum(lab for _, lab in allv[i:j])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def build():
    rnd = random.Random(7)
    M, H, R = _docs("machine"), _docs("human", "genre"), _docs("ai", "model")
    print("docs: machine", len(M), "human", len(H), "raw model", len(R), file=sys.stderr)

    def df(docs):
        c = Counter()
        for d in docs:
            c.update({w for s in d["sents"] for w in s})
        return c
    groups = sorted({d["group"] for d in M})
    dfs = [(df([d for d in M if d["group"] == g]), sum(1 for d in M if d["group"] == g)) for g in groups] + [(df(H), len(H))]
    vocab = {w for w in dfs[-1][0] if all(c.get(w, 0) / n >= 0.05 for c, n in dfs)}
    for docs in (M, H):
        rnd.shuffle(docs)
    cut_m, cut_h = int(len(M) * 0.85), int(len(H) * 0.85)

    def sents(docs):
        return [mask(s, vocab) for d in docs for s in d["sents"]]
    tm, th = sents(M[:cut_m]), sents(H[:cut_h])
    budget = min(sum(map(len, tm)), sum(map(len, th)))            # same amount of text for both models

    def take(ss):
        out, n = [], 0
        for s in ss:
            out.append(s)
            n += len(s)
            if n >= budget:
                break
        return out
    model = {"vocab": vocab, "machine": train(take(tm)), "human": train(take(th))}
    model["V"] = len(set(model["machine"]["c"][1]) | set(model["human"]["c"][1])) + 1

    def scores(docs):
        return [v for d in docs for s in d["sents"] if (v := llr(model, s)) is not None]
    sm, sh, sr = scores(M[cut_m:]), scores(H[cut_h:]), scores(R)
    sh_sorted = sorted(sh)
    model["threshold"] = sh_sorted[int(len(sh_sorted) * 0.97)]     # 3% of held-out human sentences exceed it
    MODEL.parent.mkdir(exist_ok=True)
    MODEL.write_bytes(pickle.dumps(model))
    flag = lambda xs: round(sum(v > model["threshold"] for v in xs) / len(xs), 3)
    print(json.dumps({"vocab": len(vocab), "train_tokens_each": budget, "threshold": round(model["threshold"], 3),
                      "held_out_sentences": {"machine": len(sm), "human": len(sh), "raw_model": len(sr)},
                      "sentence_auc": {"machine_vs_human": round(_auc(sm, sh), 3), "raw_model_vs_human": round(_auc(sr, sh), 3)},
                      "flagged_share": {"human": flag(sh), "machine": flag(sm), "raw_model": flag(sr)},
                      "model_mb": round(MODEL.stat().st_size / 2**20, 1)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    build()
