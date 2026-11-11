"""The agent: one turn in, a stream of events out.

  route     decision model (route/jev.py) when configured, otherwise a rule
  chat      reaction skill picks a meme or none -> context -> chat model -> reply (+ image)
  write     memory recalls related turns -> frontier model drafts -> style gate -> flagged sentences rewritten
Events are documented in web/server.py.
"""
import queue
import threading

from gate.check import check, mark
from . import chat, context, llm, writer
from .skills import memes

PLAIN_SYSTEM = "You are a helpful assistant."


def pieces(text, n=3):
    return [text[i:i + n] for i in range(0, len(text), n)]


def turn(history: list[dict]):
    yield {"type": "thinking", "label": "讀訊息"}
    to, by = chat.route(history)
    if to == "write":
        yield from _write(history, by)
        return
    text, model, reaction = chat.reply(history)
    yield {"type": "route", "to": "chat", "model": model, "by": by}
    for p in pieces(text):
        yield {"type": "delta", "text": p}
    yield {"type": "message_end", "meme": reaction, "model": model}


def _write(history, by):
    user = history[-1]["content"]
    yield {"type": "route", "to": "write", "model": llm.WRITE_MODEL, "by": by}
    text, chat_model = chat.ack(history)
    for p in pieces(text):
        yield {"type": "delta", "text": p}
    yield {"type": "message_end", "meme": None, "model": chat_model}
    yield {"type": "artifact_open"}

    out = queue.Queue()

    def baseline():                      # the same request with no control, for the comparison tab
        try:
            t = llm.claude(PLAIN_SYSTEM, [{"role": "user", "content": user}], llm.WRITE_MODEL)
            r = check(t)
            out.put({"type": "baseline", "text": t, "marked": mark(t, r["hits"], r["flow"]), "check": writer._public(r)})
        except Exception:  # noqa: BLE001 - optional
            pass

    def controlled():
        try:
            out.put({"type": "step", "id": "draft", "label": "起草", "state": "start"})
            for ev in writer.run(context.writing(history)):
                n = ev["check"]["flagged"] + ev["check"]["flow"]
                if ev["type"] == "draft":
                    out.put({"type": "step", "id": "draft", "label": "起草", "state": "ok"})
                    out.put({"type": "step", "id": "check", "label": f"檢查用詞 · {n} 處", "state": "ok"})
                elif ev["type"] == "repair":
                    out.put({"type": "step", "id": f"r{ev['round']}", "label": f"改寫第 {ev['round']} 輪 · 剩 {n} 處", "state": "ok"})
                out.put(ev)
        except Exception as e:  # noqa: BLE001
            out.put({"type": "error", "text": repr(e)[:300]})

    workers = [threading.Thread(target=f, daemon=True) for f in (baseline, controlled)]
    for w in workers:
        w.start()
    while any(w.is_alive() for w in workers) or not out.empty():
        try:
            yield out.get(timeout=0.2)
        except queue.Empty:
            pass
