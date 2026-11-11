"""Run the demo script through the live server once and save every event it produced.

    python tools/record_demo.py            # server must be running on :8787
The recording (demo/recording.json) holds real outputs. Start the server with WW_REPLAY=demo/recording.json
to play the same turns back at a watchable pace (generation takes 20-40 s per article).
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
script = json.loads((ROOT / "demo" / "script.json").read_text())
history, turns = [], []
for user in script:
    history.append({"role": "user", "content": user})
    req = urllib.request.Request("http://127.0.0.1:8787/api/chat", json.dumps({"messages": history, "live": True}).encode(),
                                 {"Content-Type": "application/json"})
    events = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                if ev["type"] != "done":
                    events.append(ev)
    for ev in events:
        if ev["type"] == "message":
            history.append({"role": "assistant", "content": ev["text"]})
        if ev["type"] == "error":
            sys.exit(f"error on {user!r}: {ev['text']}")
    turns.append({"user": user, "events": events})
    print(user, "->", [e["type"] for e in events], flush=True)
(ROOT / "demo" / "recording.json").write_text(json.dumps({"turns": turns}, ensure_ascii=False, indent=1))
