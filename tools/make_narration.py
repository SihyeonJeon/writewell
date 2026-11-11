"""Synthesize the narration segments (ElevenLabs) and record each one's length.

    python tools/make_narration.py      # needs ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID in .env
Writes out/audio/<id>.mp3 and adds "seconds" to demo/narration.json. A segment is re-made only when its text changed.
"""
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for line in (ROOT / ".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
out = ROOT / "out" / "audio"
out.mkdir(parents=True, exist_ok=True)
f = ROOT / "demo" / "narration.json"
segs = json.loads(f.read_text())
for s in segs:
    h = hashlib.md5((s["text"] + os.environ["ELEVENLABS_VOICE_ID"]).encode()).hexdigest()[:10]
    mp3 = out / f"{s['id']}-{h}.mp3"
    if not mp3.exists():
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{os.environ['ELEVENLABS_VOICE_ID']}",
            json.dumps({"text": s["text"], "model_id": "eleven_multilingual_v2",
                        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "speed": 1.08}}).encode(),
            {"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Content-Type": "application/json"})
        mp3.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    s["audio"] = str(mp3.relative_to(ROOT))
    s["seconds"] = round(float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp3)]).decode()), 2)
    print(s["id"], s["seconds"], "s", len(s["text"].split()), "words")
f.write_text(json.dumps(segs, ensure_ascii=False, indent=1))
print("total", round(sum(s["seconds"] for s in segs), 1), "s")
