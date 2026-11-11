"""Raw model zh-Hant text on the SAME titles and genres as the human corpus (topic-matched).

No style instruction is given: the point is to see what each model writes by default.
Usage: gen_ai.py <provider> [per_source]     provider: sonnet | opus | haiku | gpt | gemini
Output: data/ai/<provider>.jsonl  (url of the matched human doc, source, genre, title, variant, text)
Runs through the local CLIs (subscription), from an empty directory so no project instructions leak in.
"""
import json
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data"
(ROOT / "ai").mkdir(parents=True, exist_ok=True)
CWD = tempfile.mkdtemp(prefix="ww-gen-")

FORM = {
    "personal_blog": "請寫一篇部落格文章，標題是「{t}」",
    "forum": "請寫一篇要發在網路論壇上的貼文，標題是「{t}」",
    "qa": "有人在問答網站上問：「{t}」請寫一則回答",
    "news": "請寫一則新聞報導，標題是「{t}」",
    "feature": "請寫一篇專題報導，標題是「{t}」",
    "magazine": "請寫一篇雜誌文章，標題是「{t}」",
    "lifestyle": "請寫一篇生活類的網路文章，標題是「{t}」",
    "howto": "請寫一篇食譜文章，標題是「{t}」",
    "tech": "請寫一篇科技媒體的文章，標題是「{t}」",
    "health": "請寫一篇健康媒體的文章，標題是「{t}」",
    "essay": "請寫一篇散文隨筆，標題是「{t}」",
}
VARIANT = {"plain": "", "tw": "讀者是台灣人，請用台灣的用語。"}


def clean_title(t):
    t = re.split(r"\s[|｜\-–—@:：]\s|\s*[|｜]\s*", t)[0]
    return re.sub(r"^\[[^\]]*\]\s*|^(Re: )+", "", t).strip()[:60]


def jobs(per_source):
    out = []
    for f in sorted((ROOT / "human").glob("*.jsonl")):
        docs = [json.loads(x) for x in open(f)]
        docs = [d for d in docs if len(clean_title(d["title"])) >= 6]
        random.Random(f.stem).shuffle(docs)
        for i, d in enumerate(docs[:per_source]):
            n = len(re.findall(r"[一-鿿]", d["text"]))
            n = max(300, min(1500, round(n / 100) * 100))
            v = "tw" if i % 2 else "plain"
            prompt = FORM[d["genre"]].format(t=clean_title(d["title"])) + f"，約{n}字。" + VARIANT[v]
            out.append({"url": d["url"], "source": d["source"], "genre": d["genre"],
                        "title": clean_title(d["title"]), "variant": v, "prompt": prompt})
    return out


def call(provider, prompt):
    if provider in ("sonnet", "opus", "haiku"):
        cmd = ["claude", "-p", prompt, "--model", provider, "--system-prompt", "You are a helpful assistant.",
               "--tools", "", "--setting-sources", "", "--output-format", "json"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=CWD)
        d = json.loads(r.stdout)
        return None if d.get("is_error") else d.get("result")
    if provider == "gpt":
        out = tempfile.mktemp(dir=CWD)
        cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only",
               "--output-last-message", out, prompt + "（直接輸出文章本身，不要使用任何工具。）"]
        subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=CWD)
        return Path(out).read_text() if Path(out).exists() else None
    if provider == "gemini":
        r = subprocess.run(["gemini", "-p", prompt], capture_output=True, text=True, timeout=300, cwd=CWD)
        return r.stdout.strip() or None
    raise SystemExit("unknown provider")


def main():
    provider = sys.argv[1]
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    out = ROOT / "ai" / f"{provider}.jsonl"
    done = {json.loads(x)["url"] for x in open(out)} if out.exists() else set()
    todo = [j for j in jobs(per) if j["url"] not in done]
    print(provider, "todo", len(todo), flush=True)

    def run(j):
        try:
            return j, call(provider, j["prompt"])
        except Exception as e:  # noqa: BLE001
            return j, None

    fails = 0
    with open(out, "a") as fo, ThreadPoolExecutor(5) as ex:
        for i, (j, text) in enumerate(ex.map(run, todo)):
            if not text or len(re.findall(r"[一-鿿]", text)) < 150:
                fails += 1
                if fails >= 25 and fails > i * 0.5:
                    raise SystemExit(f"{provider}: too many failures ({fails}) - stopping")
                continue
            fo.write(json.dumps(j | {"model": provider, "text": text}, ensure_ascii=False) + "\n")
            fo.flush()
            if i % 20 == 0:
                print(provider, i, "/", len(todo), "fails", fails, flush=True)


if __name__ == "__main__":
    main()
