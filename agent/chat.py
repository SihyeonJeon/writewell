"""Conversation side: persona, deterministic routing, meme attachment.

Routing is a plain rule for now: a request to write something goes to the controlled writer, everything else
to the chat model. route/jev.py holds the decision-model version that will replace the rule.
"""
import json
import re
from pathlib import Path

from . import llm

ROOT = Path(__file__).resolve().parent.parent
MEMES = ROOT / "memes"

WRITE_RE = re.compile(
    r"寫(一|個|篇|封|段|首|則|點|份|下)|撰寫|起草|擬一|write\s+(me\s+)?(a|an|the|some)\b|draft\s+(a|an|the)\b", re.I)

PERSONA = """你叫小寫，是住在台北的朋友，跟對方用 LINE 聊天的那種口氣。
一次只講一兩句，像真人打字：會用「啦」「喔」「欸」「吧」「齁」，會用「真的」「超」「蠻」，偶爾「XD」或「～」。
有自己的喜好跟脾氣，會吐槽，也會接梗。對方心情不好就陪他講，別急著給辦法。
用台灣的詞。別條列，別粗體，別用書面語，別在句尾問對方還需要什麼。
對方要你寫文章、貼文、信之類的東西時，只回一句話說你去寫了，文章會出現在旁邊的面板。"""


def route(message: str) -> str:
    return "write" if WRITE_RE.search(message) else "chat"


def memes():
    f = MEMES / "manifest.json"
    return json.loads(f.read_text()) if f.exists() else []


def pick_meme(user: str, reply: str, last_had_meme: bool):
    """At most one image, never twice in a row. A meme entry lists the situations (regex) it fits."""
    if last_had_meme:
        return None
    for m in memes():
        if any(re.search(p, user) or re.search(p, reply) for p in m.get("when", [])):
            return {"url": "/memes/" + m["file"], "alt": m.get("alt", ""), "id": m["id"]}
    return None


def reply(history: list[dict]) -> tuple[str, str]:
    return llm.chat(PERSONA, history[-12:])


def ack(history: list[dict]) -> tuple[str, str]:
    """One line saying the piece is on its way. Anything past the first line is dropped."""
    text, model = llm.chat(PERSONA, history[-6:])
    return text.strip().split("\n")[0][:80], model
