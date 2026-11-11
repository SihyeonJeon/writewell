"""Demo server: chat panel + writing panel. Standard library only.

Run:  python -m web.server   (http://localhost:8787)
Env:  CHAT_BASE_URL / CHAT_MODEL   OpenAI-compatible endpoint for the fine-tuned Gemma (optional)
      ANTHROPIC_API_KEY            optional; without it the local `claude` CLI is used
"""
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent import chat, llm, writer
from gate.check import check, mark

HERE = Path(__file__).parent
PLAIN_SYSTEM = "You are a helpful assistant."


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _file(self, path, ctype=None):
        if not path.is_file():
            return self.send_error(404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._file(HERE / "index.html", "text/html; charset=utf-8")
        if self.path.startswith("/memes/"):
            name = Path(self.path).name
            return self._file(chat.MEMES / name)
        self.send_error(404)

    def _event(self, obj):
        self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
        self.wfile.flush()

    def do_POST(self):
        if self.path != "/api/chat":
            return self.send_error(404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        history, last_meme = body["messages"], body.get("last_had_meme", False)
        user = history[-1]["content"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            if chat.route(user) == "write":
                self._write(history, user)
            else:
                text, model = chat.reply(history)
                self._event({"type": "route", "to": "chat", "model": model})
                self._event({"type": "message", "text": text, "meme": chat.pick_meme(user, text, last_meme)})
        except Exception as e:  # noqa: BLE001 - show the failure in the UI instead of a hung stream
            self._event({"type": "error", "text": repr(e)[:300]})
        self._event({"type": "done"})

    def _write(self, history, user):
        self._event({"type": "route", "to": "write", "model": llm.WRITE_MODEL + " + style gate"})
        self._event({"type": "artifact_open"})
        lock = threading.Lock()

        def side(fn):
            try:
                ev = fn()
                with lock:
                    self._event(ev)
            except Exception:  # noqa: BLE001 - side panels are optional
                pass

        def ack():
            text, _ = chat.ack(history)
            return {"type": "message", "text": text, "meme": None}

        def baseline():
            text = llm.claude(PLAIN_SYSTEM, [{"role": "user", "content": user}], llm.WRITE_MODEL)
            r = check(text)
            return {"type": "baseline", "text": text, "marked": mark(text, r["hits"]),
                    "check": writer._public(r)}

        threads = [threading.Thread(target=side, args=(f,), daemon=True) for f in (ack, baseline)]
        for t in threads:
            t.start()
        for ev in writer.run(user):
            with lock:
                self._event(ev)
        for t in threads:
            t.join(timeout=240)


if __name__ == "__main__":
    print("http://localhost:8787")
    ThreadingHTTPServer(("127.0.0.1", 8787), Handler).serve_forever()
