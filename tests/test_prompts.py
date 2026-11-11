"""No instruction sent to a model may contain a gated word or pattern."""
from agent import chat, writer
from gate.check import check


def _hits(text):
    return [h["gram"] for h in check(text)["hits"] if h["kind"] == "ban"]


def test_writer_system_has_no_banned_terms():
    assert _hits(writer.SYSTEM) == []


def test_repair_instruction_has_no_banned_terms():
    assert _hits(writer.REPAIR.format(shape=writer.SHAPE_FLAT + writer.SHAPE_Q, text="")) == []


def test_persona_has_no_banned_terms():
    assert _hits(chat.PERSONA) == []


def test_route_is_deterministic():
    assert chat.route("幫我寫一篇關於珍珠奶茶的文章") == "write"
    assert chat.route("可以寫封信給房東嗎") == "write"
    assert chat.route("write me a post about night markets") == "write"
    assert chat.route("今天好累喔") == "chat"
    assert chat.route("你覺得我該不該換工作") == "chat"


def test_gate_flags_and_marks():
    r = check("出發前先確認時間，而不是到了才問。")
    grams = {h["gram"] for h in r["hits"]}
    assert "確認" in grams and "而 不是" in grams
