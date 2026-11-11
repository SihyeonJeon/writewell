"""Controlled writing: draft with Opus, check against the gate, rewrite only what was flagged.

The instructions below never name a gated word (tests/test_prompts.py enforces it). They describe form:
what human text measurably does that machine text does not.
"""
import re

from gate.check import check, mark, surface
from . import llm

MAX_ROUNDS = 4

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


LINES = """這篇文章裡有幾句讀起來像機器寫的。我把那幾句列在最下面，像機器的字詞用【】框起來；前面標〔整句〕的，是整句的搭法像機器。

每一句都整句重寫：換個講法，意思留著，要能接回原文的上下文。【】裡的字詞別再用，也別換成意思相近的公文腔。標〔整句〕的把句子骨架換掉，像跟朋友講這件事那樣寫。

照編號輸出，一行一句，只輸出改好的句子。

原文：
{text}

要改的句子：
{lines}"""


def _marked_sentence(sent, spans, whole):
    for sp in sorted(spans, key=len, reverse=True):
        sent = re.sub(re.escape(sp) + r"(?!】)", "【" + sp + "】", sent)
    return ("〔整句〕" if whole else "") + sent.strip()


def _repair_sentences(text, result):
    """Rewrite only the flagged sentences and splice them back. A rewrite is kept only if it is clean itself."""
    sents = result["sentences"]
    spans = {}
    for h in result["hits"]:
        for si in h["sentences"]:
            spans.setdefault(si, set()).add(surface(h["gram"]))
    idx = sorted(set(spans) | set(result["flow"]))
    if not idx:
        return text
    lines = "\n".join(f"{n}. {_marked_sentence(sents[si], spans.get(si, ()), si in result['flow'])}" for n, si in enumerate(idx, 1))
    out = llm.write(SYSTEM, [{"role": "user", "content": LINES.format(text=text, lines=lines)}])
    for m in re.finditer(r"(?m)^\s*(\d+)[.、．)]\s*(.+)$", out):
        n, new = int(m.group(1)), m.group(2).strip().replace("【", "").replace("】", "").replace("〔整句〕", "")
        if not 1 <= n <= len(idx):
            continue
        still = any(sp in new for sp in spans.get(idx[n - 1], ()))          # the flagged wording must be gone
        if not still and not any(h["kind"] == "ban" for h in check(new)["hits"]):
            old = sents[idx[n - 1]]
            lead = old[:len(old) - len(old.lstrip())]
            text = text.replace(old, lead + new, 1)
    return text


def run(request: str):
    """Yields events: draft -> (repair)* -> final. Each carries the text and its gate result."""
    text = llm.write(SYSTEM, [{"role": "user", "content": request}])
    result = check(text)
    yield {"type": "draft", "round": 0, "text": text, "marked": mark(text, result["hits"], result["flow"]), "check": _public(result)}
    best, best_n = (text, result), _cost(result)
    for rnd in range(1, MAX_ROUNDS + 1):
        if result["ok"]:
            break
        if result["hits"] or result["flow"]:
            text = _repair_sentences(text, result)
        else:                                   # only the overall shape is off: one pass over the whole text
            prompt = REPAIR.format(shape=_shape_note(result["shape"]), text=text)
            text = llm.write(SYSTEM, [{"role": "user", "content": prompt}]).replace("【", "").replace("】", "").replace("〔", "").replace("〕", "")
        result = check(text)
        if _cost(result) <= best_n:
            best, best_n = (text, result), _cost(result)
        stalled = _cost(result) >= _cost(best[1]) and (text, result) != best
        text, result = best                      # never continue from a worse version
        yield {"type": "repair", "round": rnd, "text": text, "marked": mark(text, result["hits"], result["flow"]), "check": _public(result)}
    text, result = best
    yield {"type": "final", "text": text, "check": _public(result)}


def _cost(r):
    return len(r["hits"]) + len(r["flow"])


def _public(r):
    return {"ok": r["ok"], "hits": [{"gram": h["gram"].replace(" ", ""), "kind": h["kind"], "count": h["count"],
                                     "allowed": h["allowed"]} for h in r["hits"]],
            "flagged": sum(h["count"] - h["allowed"] for h in r["hits"]), "flow": len(r["flow"]), "shape": r["shape"], "tokens": r["tokens"]}
