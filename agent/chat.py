"""Conversation side: persona, deterministic routing, meme attachment.

Routing is a plain rule for now: a request to write something goes to the controlled writer, everything else
to the chat model. route/jev.py holds the decision-model version that will replace the rule.
"""
import json
import re
from pathlib import Path

from . import context, llm
from .skills import memes

ROOT = Path(__file__).resolve().parent.parent
MEMES = ROOT / "memes"

WRITE_RE = re.compile(
    r"寫(一|個|篇|封|段|首|則|點|份|下)|撰寫|起草|擬一|write\s+(me\s+)?(a|an|the|some)\b|draft\s+(a|an|the)\b", re.I)

PERSONA = context.PERSONA


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
    """-> (text, model label, reaction or None).

    The reaction skill picks a card or none. An instructed model is handed the card's line and says it;
    a fine-tuned model answers on its own, and the reaction rides along as an image (or a line when the
    card is text-only).
    """
    card = memes.pick(history[-1]["content"])
    style = context.chat_style()
    system, window = context.chat(history, card, style)
    text, model = llm.chat(system, window)
    text = text.strip()
    reaction = None
    if card:
        if style == "instructed" and card["say"] and card["say"][:6] not in text:
            text = card["say"]                # the skill's line wins when the model wandered off it
        caption = card["say"] if style == "finetuned" and not card["image"] else None
        if card["image"] or caption:
            reaction = {"url": card["image"], "alt": card["alt"], "id": card["id"], "caption": caption}
    return text, model, reaction


ACK = "好，我去寫，等等開在右邊。"


def ack(history: list[dict]) -> tuple[str, str]:
    """One line saying the piece is on its way; the model is handed the line, and the line wins if it wanders."""
    if context.chat_style() == "finetuned":      # the hand-off line is the agent's, not the chat model's
        return ACK, "agent"
    text, model = llm.chat(PERSONA + "\n\n對方要你寫東西。照樣回這一句，別多加：「" + ACK + "」", history[-4:])
    text = text.strip().split("\n")[0]
    return (text if ACK[:5] in text and len(text) < 40 else ACK), model
