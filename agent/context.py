"""What each model sees on a turn.

Two chat styles, because the two kinds of chat model need different things:
  instructed   a general model: persona as the system prompt, plus the reaction skill's line for this turn
  finetuned    our fine-tuned model was trained on bare turns (a person's message -> a person's reply), with no
               instructions anywhere. It gets the conversation window and nothing else; anything written as an
               instruction would be read as part of the message it is replying to.
The writer gets the request plus whatever earlier turns are about the same thing (agent/memory.py).
"""
import os

from . import memory
from .skills import memes

WINDOW = 8          # turns of conversation an instructed chat model sees
FINETUNED_WINDOW = int(os.environ.get("FINETUNED_WINDOW", "1"))

PERSONA = """你叫小寫，是住在台北的朋友，跟對方用 LINE 聊天的那種口氣。
一次只講一兩句，像真人打字：會用「啦」「喔」「欸」「吧」「齁」，會用「真的」「超」「蠻」，偶爾「XD」或「～」。
有自己的喜好跟脾氣，會吐槽，也會接梗。對方心情不好就陪他講，別急著給辦法。
用台灣的詞。別條列，別粗體，別用書面語，別在句尾問對方還需要什麼。
對方要你寫文章、貼文、信之類的東西時，只回一句話說你去寫了，文章會出現在旁邊的面板。"""


def chat_style() -> str:
    explicit = os.environ.get("CHAT_STYLE")
    if explicit:
        return explicit
    return "finetuned" if os.environ.get("CHAT_BASE_URL") or os.environ.get("CHAT_ANSWERS_FILE") else "instructed"


def chat(history: list[dict], card=None, style=None):
    """-> (system or None, messages) for the chat model."""
    window = history[-WINDOW:]
    while window and window[0]["role"] != "user":          # a window always opens on the user's turn
        window = window[1:]
    if (style or chat_style()) == "finetuned":
        # Measured on checkpoint-2100: with earlier turns in the window the model repeats its previous reply
        # or echoes the user (it only ever saw single exchanges). It answers the current message alone.
        return None, window[-FINETUNED_WINDOW:]
    return PERSONA + ("\n\n" + memes.hint(card) if card else ""), window


def writing(history: list[dict]) -> str:
    """The writing request, with earlier turns on the same subject as background."""
    request = history[-1]["content"]
    notes = memory.recall(history[:-1], request)
    if not notes:
        return request
    return "聊天時對方提過：\n" + "\n".join("・" + n for n in notes) + "\n\n現在他說：" + request
