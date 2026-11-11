"""Controlled writing: draft with Opus, check against the gate, rewrite only what was flagged.

The instructions below never name a gated word (tests/test_prompts.py enforces it). They describe form:
what human text measurably does that machine text does not.
"""
from gate.check import check, mark
from . import llm

MAX_ROUNDS = 3

SYSTEM = """你在台灣寫了很多年的專欄跟部落格，讀者是台灣人。照台灣人平常會讀到的文章口吻來寫。

寫法：
句子的長度要有落差。有的句子只有幾個字。有的一口氣寫得很長，中間用逗號帶著讀者走到底，講完一整個來龍去脈才收。別讓每一句都差不多長。
多寫原因、經過、結果，把事情像講故事那樣串起來，常用「所以」「因為」「結果」「的時候」「後來」這種連接。
少叫讀者去做什麼。你是在講一件事，沒有在發指示。
整篇裡問句很少，小標也別寫成問句。
全文是一段一段的散文：沒有條列、沒有粗體、沒有編號，結尾也沒有總結或提醒。
用台灣的詞：影片、品質、資訊、軟體、網路、計程車。

只輸出文章本身。第一行是標題，空一行，然後是內文。"""

REPAIR = """下面這篇文章有些地方讀起來像機器寫的，我用【】把那些字詞框起來了。

把含有【】的句子整句重寫，換個講法，意思留著。【】裡的字詞在全文都別再出現，也別換成意思相近的公文腔。
用〔〕框起來的是整句的搭法像機器：把那一句拆開重講，換掉句子的骨架，像跟朋友講這件事那樣寫。
沒有被框到的句子一個字都別動。{shape}

只輸出改好的全文，裡面沒有【】跟〔〕。

---
{text}"""

SHAPE_FLAT = "\n另外，句子長度太平均了：挑兩三處把相鄰的句子併成一個很長的句子，再留幾句只有幾個字的。"
SHAPE_Q = "\n另外，問句太多：把大部分問句改成直述句。"


def _shape_note(sh):
    return (SHAPE_FLAT if sh.get("flat") else "") + (SHAPE_Q if sh.get("too_many_questions") else "")


def run(request: str):
    """Yields events: draft -> (repair)* -> final. Each carries the text and its gate result."""
    text = llm.write(SYSTEM, [{"role": "user", "content": request}])
    result = check(text)
    yield {"type": "draft", "round": 0, "text": text, "marked": mark(text, result["hits"], result["flow"]), "check": _public(result)}
    best, best_n = (text, result), _cost(result)
    for rnd in range(1, MAX_ROUNDS + 1):
        if result["ok"]:
            break
        prompt = REPAIR.format(shape=_shape_note(result["shape"]), text=mark(text, result["hits"], result["flow"]))
        text = llm.write(SYSTEM, [{"role": "user", "content": prompt}]).replace("【", "").replace("】", "").replace("〔", "").replace("〕", "")
        result = check(text)
        if _cost(result) <= best_n:
            best, best_n = (text, result), _cost(result)
        yield {"type": "repair", "round": rnd, "text": text, "marked": mark(text, result["hits"], result["flow"]), "check": _public(result)}
    text, result = best
    yield {"type": "final", "text": text, "check": _public(result)}


def _cost(r):
    return len(r["hits"]) + len(r["flow"])


def _public(r):
    return {"ok": r["ok"], "hits": [{"gram": h["gram"].replace(" ", ""), "kind": h["kind"], "count": h["count"],
                                     "allowed": h["allowed"]} for h in r["hits"]],
            "flagged": sum(h["count"] - h["allowed"] for h in r["hits"]), "flow": len(r["flow"]), "shape": r["shape"], "tokens": r["tokens"]}
