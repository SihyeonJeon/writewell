"""Reaction skill: answer with a meme that fits the moment.

The library (memes/library/<name>/catalog.json + images) is produced by the social-monitoring collector
(collect/ig_trends.py): each card records the phrase, what it means, when it fits, and when it does not.

Selection, in order:
  1. pins   memes/pins.json - situations where a specific card must come out (regex on the user's message)
  2. tags   a card's zh tags appearing in the user's message
A picked card gives the chat model its catchphrase as a hint, and attaches the image when the card has one.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

MEMES = Path(__file__).resolve().parent.parent.parent / "memes"
LIBRARY = MEMES / "library" / "taiwan-v1"


@lru_cache
def cards():
    f = LIBRARY / "catalog.json"
    return {c["id"]: c for c in json.loads(f.read_text())["cards"]} if f.exists() else {}


@lru_cache
def pins():
    f = MEMES / "pins.json"
    return json.loads(f.read_text()) if f.exists() else []


def pick(user: str):
    """-> {"id", "phrase", "say", "image": url|None, "alt"} or None"""
    lib = cards()
    for p in pins():
        if p["card"] in lib and re.search(p["when"], user):
            return _use(lib[p["card"]], p)
    for c in lib.values():
        if any(re.search(r"[一-鿿]", t) and t in user for t in c.get("tags", [])):
            return _use(c, {})
    return None


def _use(card, pin):
    image = pin.get("image", (card.get("image_files") or [None])[0]) if pin.get("send_image", True) else None
    return {"id": card["id"], "phrase": card["title"], "say": pin.get("say"),
            "image": f"/memes/library/{LIBRARY.name}/{image}" if image else None, "alt": card["title"]}


def hint(card) -> str:
    """One line added to the chat model's instructions for this turn."""
    line = card["say"] or card["phrase"]
    return f"這一句很適合現在的情境，照樣回，別多加解釋：「{line}」"
