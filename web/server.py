"""Web agent server: chat + writing panel. Standard library only.

Run:  python -m web.server            http://localhost:8787
Config comes from the environment or a .env file in the repo root (see .env.example).

Event stream (SSE) for one turn:
  thinking {label}                     the agent is working; label says on what
  route    {to, model, by}             which model takes the turn and who decided
  delta    {text}                      chat reply, piece by piece
  message_end {meme, model}            reply finished; optional reaction image
  artifact_open                        a writing panel opens
  step     {id, label, state}          writing pipeline progress (start | ok)
  draft | repair | final {text, marked, check, round}
  baseline {text, marked, check}       the same request without any control, for comparison
  error {text} · done
"""
import json
import mimetypes
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent


def load_env():
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


load_env()
from agent import chat, core  # noqa: E402  (after the environment is loaded)

# Pacing (seconds before an event) when WW_REPLAY plays back a recording of real outputs.
PACE = {"thinking": 0.25, "route": 0.9, "delta": 0.028, "message_end": 0.15, "artifact_open": 0.3, "step": 0.9,
        "draft": 0.6, "repair": 0.5, "final": 0.5, "baseline": 0.4}


def recorded(user):
    path = os.environ.get("WW_REPLAY")
    if not path or not (ROOT / path).exists():
        return None
    for t in json.loads((ROOT / path).read_text())["turns"]:
        if t["user"] == user:
            return t["events"]
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.path = self.path.split("?")[0]
        if self.path in ("/", "/index.html"):
            return self._send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/script":
            return self._send((ROOT / "demo" / "script.json").read_bytes(), "application/json")
        if self.path.startswith("/memes/"):
            target = (chat.MEMES / self.path[len("/memes/"):]).resolve()
            if chat.MEMES.resolve() in target.parents and target.is_file():
                return self._send(target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_error(404)

    def _event(self, obj):
        self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
        self.wfile.flush()

    def do_POST(self):
        if self.path != "/api/chat":
            return self.send_error(404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        history = body["messages"]
        user = history[-1]["content"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            rec = None if body.get("live") else recorded(user)
            if rec:
                for ev in rec:
                    time.sleep(PACE.get(ev["type"], 0.2))
                    self._event(ev)
            else:
                for ev in core.turn(history):
                    self._event(ev)
            self._event({"type": "done"})
        except (BrokenPipeError, ConnectionResetError):
            return                                   # the reader pressed stop
        except Exception as e:  # noqa: BLE001 - show the failure in the UI instead of a hung stream
            self._event({"type": "error", "text": repr(e)[:300]})
            self._event({"type": "done"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8787"))
    print(f"http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
