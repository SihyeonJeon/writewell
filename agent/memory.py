"""Conversation memory: which earlier things the user said bear on what they are asking now.

No model call. Earlier user turns are scored against the current message on shared content characters and
pairs (everyday ones removed with the same list the reaction skill measured on forum text); the best few
above a floor come back, oldest first.
"""
from .skills import memes

FLOOR, KEEP = 0.08, 3


def recall(earlier: list[dict], current: str) -> list[str]:
    stop = frozenset(memes.index().get("stop", []))
    q = set(memes.grams(current, stop))
    if not q:
        return []
    scored = []
    for i, m in enumerate(earlier):
        if m["role"] != "user":
            continue
        g = set(memes.grams(m["content"], stop))
        if g and any(len(x) > 1 for x in q & g) and len(q & g) / len(q | g) >= FLOOR:      # shares a content pair
            scored.append((len(q & g) / len(q | g), i, m["content"]))
    top = sorted(scored, reverse=True)[:KEEP]
    return [text for _, _, text in sorted(top, key=lambda t: t[1])]
