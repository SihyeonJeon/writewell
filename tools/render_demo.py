"""Record the autoplayed demo with captions and lay the narration under it.

    python tools/render_demo.py ~/Downloads/writewell-demo.mp4
Needs the server running in playback mode (WW_REPLAY=demo/recording.json) and tools/make_narration.py run first.
Narration segments start at the moments the page reported for them, so voice and picture stay together.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
final = Path(sys.argv[1]).expanduser()
screen = ROOT / "out" / "video" / "screen.mp4"
subprocess.run([sys.executable, str(ROOT / "tools" / "capture_demo.py"), str(screen), "http://127.0.0.1:8787/?play&captions"], check=True)
meta = json.loads(screen.with_suffix(".cues.json").read_text())
segs = {s["id"]: s for s in json.loads((ROOT / "demo" / "narration.json").read_text())}
cmd, filt, mix = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(screen)], [], []
for i, c in enumerate(meta["cues"], 1):
    cmd += ["-i", str(ROOT / segs[c["id"]]["audio"])]
    ms = max(int((c["epoch"] - meta["t0"]) * 1000), 0)
    filt.append(f"[{i}:a]adelay={ms}:all=1[a{i}]")
    mix.append(f"[a{i}]")
    print(c["id"], "at", round(ms / 1000, 2), "s for", segs[c["id"]]["seconds"], "s")
filt.append("".join(mix) + f"amix=inputs={len(mix)}:normalize=0:duration=longest[a]")
final.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(cmd + ["-filter_complex", ";".join(filt), "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                      "-movflags", "+faststart", str(final)], check=True)
dur = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(final)]).decode().strip()
print(final, dur, "s")
