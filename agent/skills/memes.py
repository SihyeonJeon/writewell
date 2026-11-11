"""Reaction skill: answer with a meme when one fits, and with none when nothing does.

The library (memes/library/<name>/) comes from the social-monitoring collector (collect/ig_trends.py):
catalog.json holds each card (phrase, meaning, when it fits, when it does not, image file).

Choosing a card must stay cheap as the library grows, so nothing reads images or cards per turn.
`build` turns every card into one small vector once (character n-gram TF-IDF over the card's phrase, the text
on the image, and short lines describing when people would use it). A turn costs one dot product per card.

  1. serious   the message is about real trouble (job loss, illness, grief...)        -> no meme
  2. pins      memes/pins.json: situations where a specific card must come out        -> that card
  3. retrieve  best card by similarity; needs a minimum score and a lead over the runner-up,
               and the card's own "avoid" cues must not match                         -> that card or none
A chosen card hands the chat model its line as a hint and attaches the image when the card has one.

    python -m agent.skills.memes build        # rebuild index.json after the library changes
    python -m agent.skills.memes "今天又加班"   # show the ranking for a message
"""
import json
import math
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

MEMES = Path(__file__).resolve().parent.parent.parent / "memes"
LIBRARY = MEMES / "library" / "taiwan-v1"
MIN_SCORE, MIN_LEAD = 0.18, 0.0


def grams(text, stop=frozenset()):
    t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text.lower())
    return Counter(g for g in [t[i:i + 2] for i in range(len(t) - 1)] + list(t) if g not in stop)


def common_grams(top_chars=150, top_pairs=400):
    """Everyday characters and pairs carry no situation. Measured on forum text when the corpus is present."""
    files = sorted((MEMES.parent / "data" / "human").glob("*.jsonl"))
    files = [f for f in files if f.stem in ("ptt", "dcard", "mobile01", "yahoo_answers", "pixnet")]
    if not files:
        return index().get("stop", [])
    one, two = Counter(), Counter()
    for f in files:
        for line in open(f):
            t = re.sub(r"[^\u4e00-\u9fff]", "", json.loads(line)["text"])[:3000]
            one.update(t)
            two.update(t[i:i + 2] for i in range(len(t) - 1))
    return [g for g, _ in one.most_common(top_chars)] + [g for g, _ in two.most_common(top_pairs)]


@lru_cache
def cards():
    f = LIBRARY / "catalog.json"
    return {c["id"]: c for c in json.loads(f.read_text())["cards"]} if f.exists() else {}


@lru_cache
def situations():
    f = LIBRARY / "situations.zh.json"
    return json.loads(f.read_text()) if f.exists() else {"serious": "(?!)", "cards": {}}


@lru_cache
def pins():
    f = MEMES / "pins.json"
    return json.loads(f.read_text()) if f.exists() else []


@lru_cache
def index():
    f = LIBRARY / "index.json"
    return json.loads(f.read_text()) if f.exists() else {"stop": [], "idf": {}, "cards": {}}


def build():
    stop = common_grams()
    sit, docs = situations()["cards"], {}
    for cid, c in cards().items():
        zh_tags = [t for t in c.get("tags", []) if re.search(r"[\u4e00-\u9fff]", t) and not re.search(r"[\uac00-\ud7a3]", t)]
        lines = sit.get(cid, {}).get("say", []) + zh_tags + [c.get("image_text") or ""] + [e["reply_zh_tw"] for e in c.get("examples", [])]
        docs[cid] = grams(" ".join(lines), frozenset(stop))
    df = Counter(g for d in docs.values() for g in d)
    idf = {g: math.log((1 + len(docs)) / (1 + n)) + 1 for g, n in df.items()}
    out = {}
    for cid, d in docs.items():
        v = {g: (1 + math.log(n)) * idf[g] for g, n in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1
        out[cid] = {g: round(x / norm, 4) for g, x in v.items()}
    (LIBRARY / "index.json").write_text(json.dumps({"stop": stop, "idf": {g: round(x, 4) for g, x in idf.items()}, "cards": out}, ensure_ascii=False))
    index.cache_clear()
    return len(out)


def rank(message):
    ix = index()
    q = {g: (1 + math.log(n)) * ix["idf"][g] for g, n in grams(message).items() if g in ix["idf"]}
    norm = math.sqrt(sum(x * x for x in q.values())) or 1
    return sorted(((sum(w * vec.get(g, 0) for g, w in q.items()) / norm, cid) for cid, vec in ix["cards"].items()), reverse=True)


def pick(user: str):
    """-> {"id", "phrase", "say", "image": url|None, "alt", "by"} or None"""
    lib, sit = cards(), situations()
    if not lib or re.search(sit["serious"], user):
        return None
    for p in pins():
        if p["card"] in lib and re.search(p["when"], user):
            return _use(lib[p["card"]], p, "pin")
    r = rank(user)
    if not r or r[0][0] < MIN_SCORE or (len(r) > 1 and r[0][0] - r[1][0] <= MIN_LEAD):
        return None
    cid = r[0][1]
    avoid = sit["cards"].get(cid, {}).get("avoid")
    if avoid and re.search(avoid, user):
        return None
    return _use(lib[cid], {}, f"retrieve {r[0][0]:.2f}")


def _use(card, pin, by):
    image = pin.get("image", (card.get("image_files") or [None])[0]) if pin.get("send_image", True) else None
    return {"id": card["id"], "phrase": card["title"], "say": pin.get("say"), "by": by,
            "image": f"/memes/library/{LIBRARY.name}/{image}" if image else None, "alt": card["title"]}


def hint(card) -> str:
    """One line added to the chat model's instructions for this turn."""
    if card["say"]:
        return f"這一句很適合現在的情境，照樣回，別多加解釋：「{card['say']}」"
    return f"現在的情境很適合用這個梗：「{card['phrase']}」。用它回一句，別解釋梗。"


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        print("cards indexed:", build())
    else:
        for score, cid in rank(sys.argv[1]):
            print(f"{score:.3f}  {cid}")
        print("pick:", pick(sys.argv[1]))
