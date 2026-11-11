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
    assert _hits(chat.ACK) == []


def test_sentence_repair_instruction_has_no_banned_terms():
    assert _hits(writer.LINES.format(text="", lines="")) == []


def test_route_is_deterministic():
    r = lambda t: chat.route([{"role": "user", "content": t}])[0]
    assert r("幫我寫一篇關於珍珠奶茶的文章") == "write"
    assert r("可以寫封信給房東嗎") == "write"
    assert r("write me a post about night markets") == "write"
    assert r("今天好累喔") == "chat"
    assert r("你覺得我該不該換工作") == "chat"


def test_gate_flags_and_marks():
    r = check("出發前先確認時間，而不是到了才問。")
    grams = {h["gram"] for h in r["hits"]}
    assert "確認" in grams and "而 不是" in grams


def test_meme_lines_have_no_banned_terms():
    from agent.skills import memes
    for p in memes.pins():
        assert _hits(p.get("say", "")) == [], p["card"]
    assert _hits(memes.hint({"say": None, "phrase": "x"})) == []


def test_pinned_situations_pick_their_card():
    from agent.skills import memes
    if not memes.cards():
        return
    assert memes.pick("欸我們年初說要去花蓮，到現在都沒人訂票")["id"] == "ven-again"
    assert memes.pick("禮拜一就開了六個會，說好的輕鬆禮拜一咧")["id"] == "holding-on-pirate"


def test_retrieval_picks_a_fitting_card_and_none_otherwise(monkeypatch):
    from agent.skills import memes
    if not memes.index()["cards"]:
        return
    monkeypatch.setattr(memes, "pins", lambda: [])
    assert memes.pick("週末加班 累到不想講話")["id"] == "tired-to-hyped"
    assert memes.pick("問了三次要不要去墾丁 沒人理")["id"] == "ven-again"
    for text in ("你覺得宇宙有盡頭嗎", "台北哪裡有好吃的牛肉麵", "今天天氣不錯欸", "幫我寫一篇揪大家去花蓮的貼文"):
        assert memes.pick(text) is None, text
    assert memes.pick("我被裁員了 明天不用去上班") is None      # real trouble: no joke
    assert memes.pick("我爸住院了 我好累") is None


def test_interface_text_has_no_banned_terms():
    import re
    from pathlib import Path
    html = Path(__file__).resolve().parent.parent.joinpath("web", "index.html").read_text()
    strings = re.findall(r"[\u4e00-\u9fff][\u4e00-\u9fff，。、…！？ ]*", re.sub(r"<style>.*?</style>", "", html, flags=re.S))
    assert _hits("。".join(strings)) == []
