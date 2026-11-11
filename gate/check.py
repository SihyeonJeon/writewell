"""Style gate: check a zh-Hant text against the measured lexicon (gate/zh-Hant.json) and shape thresholds.

check(text) -> {"ok", "hits": [{"gram", "kind", "sentence", "count", "allowed"}], "shape": {...}, "sentences": [...]}
Matching is on jieba tokens, the same segmentation the statistics were computed with.
"""
import json
import math
import re
import statistics
from functools import lru_cache
from pathlib import Path

import jieba
from opencc import OpenCC

jieba.setLogLevel(60)
_T2S = OpenCC("t2s")
_LAT, _NUM = "Ａ", "９"
_PUNCT = "，。！？；：、「」『』（）《》…—"
_KEEP = re.compile(rf"[^一-鿿㐀-䶿{_LAT}{_NUM}{_PUNCT}]")
_H2F = str.maketrans({",": "，", "!": "！", "?": "？", ";": "；", ":": "：", "(": "（", ")": "）"})


@lru_cache
def lexicon(lang="zh-Hant"):
    d = json.loads((Path(__file__).parent / f"{lang}.json").read_text())
    return d["ban"], d["cap"], d["shape"]


def tokens(sentence: str) -> list[str]:
    t = re.sub(r"https?://\S+", " ", sentence)
    t = re.sub(r"\d[\d,.\-~％%/:]*", _NUM, t)
    t = re.sub(r"[A-Za-z][A-Za-z'’.&+\- ]*", _LAT, t)
    t = _KEEP.sub("", t.translate(_H2F))
    s = _T2S.convert(t)
    if len(s) != len(t):
        return list(jieba.cut(t, HMM=False))
    out, i = [], 0
    for w in jieba.cut(s, HMM=False):
        out.append(t[i:i + len(w)])
        i += len(w)
    return out


def split(text: str) -> list[str]:
    """Raw sentences, in order. Joining them with the original separators is not needed: repair works on whole text."""
    out = []
    for line in text.split("\n"):
        out += [s for s in re.findall(r"[^。！？!?；]+[。！？!?；]*", line) if s.strip()]
    return out


def check(text: str, lang="zh-Hant") -> dict:
    ban, cap, shape = lexicon(lang)
    sents = split(text)
    toks = [tokens(s) for s in sents]
    total = sum(len(t) for t in toks) or 1
    found = {}                                   # gram -> [sentence index, ...]
    grams = set(ban) | set(cap)
    for si, t in enumerate(toks):
        for n in (1, 2, 3):
            for i in range(len(t) - n + 1):
                g = " ".join(t[i:i + n])
                if g in grams:
                    found.setdefault(g, []).append(si)
    hits = []
    for g, where in found.items():
        if g in ban:
            hits.append({"gram": g, "kind": "ban", "sentences": sorted(set(where)), "count": len(where), "allowed": 0})
        else:
            allowed = max(1, math.ceil(cap[g]["allow_per1k"] * total / 1000))
            if len(where) > allowed:
                hits.append({"gram": g, "kind": "cap", "sentences": sorted(set(where[allowed:])), "count": len(where), "allowed": allowed})
    lens = [len(re.findall(r"[一-鿿]", s)) for s in sents]
    lens = [n for n in lens if n >= 4]
    sh = {"sentences": len(lens)}
    if len(lens) >= 8:
        sh["len_sd"] = round(statistics.pstdev(lens), 1)
        sh["long_share"] = round(sum(n > 60 for n in lens) / len(lens), 3)
        sh["question_per1k"] = round(1000 * len(re.findall(r"[？?]", text)) / max(len(text), 1), 2)
        sh["flat"] = sh["len_sd"] < shape["sentence_len_sd_min"]
        sh["too_many_questions"] = sh["question_per1k"] > shape["question_per1k_chars_max"]
    ok = not hits and not sh.get("flat") and not sh.get("too_many_questions")
    return {"ok": ok, "hits": hits, "shape": sh, "sentences": sents, "tokens": total}


def surface(gram: str) -> str:
    return gram.replace(" ", "")


def mark(text: str, hits: list[dict], open_="【", close="】") -> str:
    """Bracket every flagged span in its sentence (used for the repair request and for the UI)."""
    sents = split(text)
    flagged = {}
    for h in hits:
        for si in h["sentences"]:
            flagged.setdefault(si, set()).add(surface(h["gram"]))
    out = text
    for si, spans in flagged.items():
        s = new = sents[si]
        for sp in sorted(spans, key=len, reverse=True):
            new = re.sub(re.escape(sp) + rf"(?!{re.escape(close)})", open_ + sp + close, new)
        out = out.replace(s, new, 1)
    return out


if __name__ == "__main__":
    import sys
    r = check(sys.stdin.read())
    print(json.dumps({k: r[k] for k in ("ok", "hits", "shape")}, ensure_ascii=False, indent=1))
