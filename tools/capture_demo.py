"""Record the autoplayed demo to a video file through Chrome's screencast (separate headless browser profile).

    python tools/capture_demo.py out/demo-screen.mp4 [url]
The page (?play) exposes window.__cues = [{id, t}] in milliseconds from the start of capture, used to place narration.
"""
import base64
import json
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import websocket

out = Path(sys.argv[1])
url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8787/?play"
W, H = 1920, 1080
tmp = Path(tempfile.mkdtemp(prefix="ww-cap-"))
chrome = subprocess.Popen(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new", "--remote-debugging-port=9334",
                           "--remote-allow-origins=*", f"--user-data-dir={tmp}/profile", f"--window-size={W},{H}", "--hide-scrollbars", "about:blank"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(60):
        try:
            tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9334/json"))
            break
        except Exception:
            time.sleep(0.25)
    ws = websocket.create_connection([t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"], timeout=30)
    lock, n, replies, frames, done = threading.Lock(), [0], {}, [], threading.Event()

    def send(method, **params):
        with lock:
            n[0] += 1
            ws.send(json.dumps({"id": n[0], "method": method, "params": params}))
            return n[0]

    def reader():
        while not done.is_set():
            try:
                m = json.loads(ws.recv())
            except Exception:
                break
            if m.get("method") == "Page.screencastFrame":
                p = m["params"]
                f = tmp / f"f{len(frames):06d}.jpg"
                f.write_bytes(base64.b64decode(p["data"]))
                frames.append((p["metadata"]["timestamp"], f))
                send("Page.screencastFrameAck", sessionId=p["sessionId"])
            elif "id" in m:
                replies[m["id"]] = m

    threading.Thread(target=reader, daemon=True).start()

    def call(method, **params):
        i = send(method, **params)
        for _ in range(600):
            if i in replies:
                return replies.pop(i)
            time.sleep(0.05)
        raise TimeoutError(method)

    call("Page.enable")
    call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
    call("Page.navigate", url=url)
    time.sleep(0.6)
    call("Page.startScreencast", format="jpeg", quality=92, everyNthFrame=1)
    t_start = time.time()
    while time.time() - t_start < 240:
        time.sleep(1)
        r = call("Runtime.evaluate", expression="window.__done === true")
        if r["result"]["result"].get("value"):
            break
    time.sleep(1.0)
    cues = call("Runtime.evaluate", expression="JSON.stringify(window.__cues || [])")["result"]["result"]["value"]
    call("Page.stopScreencast")
    done.set()
    # frames arrive only when the picture changes: hold each one until the next
    t0 = frames[0][0]
    lst = tmp / "frames.txt"
    with open(lst, "w") as f:
        for (t, p), (t2, _) in zip(frames, frames[1:] + [(frames[-1][0] + 1.0, None)]):
            f.write(f"file '{p}'\nduration {max(t2 - t, 0.001):.4f}\n")
        f.write(f"file '{frames[-1][1]}'\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-vf", "fps=30,format=yuv420p",
                    "-c:v", "libx264", "-crf", "18", "-preset", "medium", str(out)], check=True)
    out.with_suffix(".cues.json").write_text(json.dumps({"t0": t0, "cues": json.loads(cues), "frames": len(frames), "seconds": round(frames[-1][0] - t0, 2)}, indent=1))
    print(out, "frames", len(frames), "seconds", round(frames[-1][0] - t0, 1), "cues", cues[:200])
finally:
    chrome.terminate()
