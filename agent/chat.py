"""Conversation side: persona, deterministic routing, meme attachment.

Routing is a plain rule for now: a request to write something goes to the controlled writer, everything else
to the chat model. route/jev.py holds the decision-model version that will replace the rule.
"""
import json
import re
from pathlib import Path

from . import llm
from .skills import memes

ROOT = Path(__file__).resolve().parent.parent
MEMES = ROOT / "memes"

WRITE_RE = re.compile(
    r"寫(一|個|篇|封|段|首|則|點|份|下)|撰寫|起草|擬一|write\s+(me\s+)?(a|an|the|some)\b|draft\s+(a|an|the)\b", re.I)

PERSONA = """你叫小寫，是住在台北的朋友，跟對方用 LINE 聊天的那種口氣。
一次只講一兩句，像真人打字：會用「啦」「喔」「欸」「吧」「齁」，會用「真的」「超」「蠻」，偶爾「XD」或「～」。
有自己的喜好跟脾氣，會吐槽，也會接梗。對方心情不好就陪他講，別急著給辦法。
用台灣的詞。別條列，別粗體，別用書面語，別在句尾問對方還需要什麼。
對方要你寫文章、貼文、信之類的東西時，只回一句話說你去寫了，文章會出現在旁邊的面板。"""


def route(history: list[dict]) -> tuple[str, str]:
    """-> ("write" | "chat", who decided). The decision model runs when its key is set; the rule is the fallback."""
    import os
    message = history[-1]["content"]
    if os.environ.get("OPENROUTER_API_KEY"):
        from route import jev
        r = jev.route(history)
        if r.reason != "fallback":
            to = "chat" if r.model == "gemma" else "write" if r.form == "longform" or WRITE_RE.search(message) else "chat"
            return to, "jev: " + r.reason
    return ("write" if WRITE_RE.search(message) else "chat"), "rule"


def reply(history: list[dict]):
    """-> (text, model label, meme or None). The reaction skill may hand the model a line and an image."""
    card = memes.pick(history[-1]["content"])
    system = PERSONA + ("\n\n" + memes.hint(card) if card else "")
    text, model = llm.chat(system, history[-12:])
    if card and card["say"] and card["say"][:6] not in text:
        text = card["say"]                    # the skill's line wins when the model wandered off it
    meme = {"url": card["image"], "alt": card["alt"], "id": card["id"]} if card and card["image"] else None
    return text.strip(), model, meme


ACK = "好，我去寫，等等開在右邊。"


def ack(history: list[dict]) -> tuple[str, str]:
    """One line saying the piece is on its way; the model is handed the line, and the line wins if it wanders."""
    text, model = llm.chat(PERSONA + "\n\n對方要你寫東西。照樣回這一句，別多加：「" + ACK + "」", history[-4:])
    text = text.strip().split("\n")[0]
    return (text if ACK[:5] in text and len(text) < 40 else ACK), model
