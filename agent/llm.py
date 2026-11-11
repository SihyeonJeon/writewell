"""Model access. Three backends behind two functions.

chat model  : any OpenAI-compatible endpoint (CHAT_BASE_URL + CHAT_MODEL) - this is where the fine-tuned Gemma goes.
              Without it, falls back to a small Claude model so the demo still runs.
write model : Claude Opus through the Anthropic API (ANTHROPIC_API_KEY), or through the local `claude` CLI when no key is set.
"""
import json
import os
import re
import subprocess
import tempfile
import urllib.request

WRITE_MODEL = os.environ.get("WRITE_MODEL", "claude-opus-5-5")
FALLBACK_CHAT_MODEL = os.environ.get("FALLBACK_CHAT_MODEL", "claude-haiku-5-5")
_CLI_ALIAS = {"claude-opus-5-5": "opus", "claude-haiku-5-5": "haiku", "claude-sonnet-5-5": "sonnet"}
_CWD = tempfile.mkdtemp(prefix="ww-")       # empty directory: no project instructions reach the CLI


def _post(url, body, headers, timeout=180):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"} | headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def claude(system: str, messages: list[dict], model: str, max_tokens=4000) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        d = _post("https://api.anthropic.com/v1/messages",
                  {"model": model, "max_tokens": max_tokens, "system": system, "messages": messages},
                  {"x-api-key": key, "anthropic-version": "2023-06-01"})
        return "".join(b.get("text", "") for b in d["content"])
    # CLI path: one prompt string; earlier turns are folded into it
    convo = "\n\n".join(("" if m["role"] == "user" else "（你先前的回覆）") + m["content"] for m in messages)
    cmd = ["claude", "-p", convo, "--model", _CLI_ALIAS.get(model, model), "--system-prompt", system,
           "--tools", "", "--setting-sources", "", "--strict-mcp-config", "--output-format", "json"]
    for _ in range(3):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=_CWD)
        d = json.loads(r.stdout)
        if d.get("is_error"):
            raise RuntimeError(str(d.get("result"))[:300])
        out, head = d["result"], d["result"][:400]
        cjk_prompt = bool(re.search(r"[\u4e00-\u9fff]", convo))
        cjk_share = len(re.findall(r"[\u4e00-\u9fff]", head)) / max(len(head), 1)
        if not _JUNK.search(head) and not (cjk_prompt and cjk_share < 0.3):
            return _LEAD.sub("", out)
    raise RuntimeError("CLI kept returning scaffolding text instead of an answer")


_LEAD = re.compile(r"^[^\w#*「《（\u4e00-\u9fff]+")      # stray leading symbols


# The CLI occasionally returns internal scaffolding (tags, rulers, notes to self) in place of the answer.
_JUNK = re.compile(r"<\/?(thinking|reasoning)[\w_]*>|═{5,}|\ufffd|^\s*(The user|User wa)|Claude|Anthropic", re.M)


def chat(system, messages: list[dict]) -> tuple[str, str]:
    """Returns (text, model label).

    CHAT_BASE_URL      live endpoint serving the fine-tuned chat model (OpenAI-compatible)
    CHAT_ANSWERS_FILE  answers collected from that model offline: {"<user message>": "<its reply>"}
    neither            a small Claude model answers so the app still runs
    """
    base, cached = os.environ.get("CHAT_BASE_URL"), os.environ.get("CHAT_ANSWERS_FILE")
    model = os.environ.get("CHAT_MODEL", "chat")
    if cached and os.path.exists(cached):
        answers = json.load(open(cached, encoding="utf-8"))
        if messages[-1]["content"] in answers:
            return answers[messages[-1]["content"]].strip(), model
    if base:
        d = _post(base.rstrip("/") + "/chat/completions",
                  {"model": model, "messages": ([{"role": "system", "content": system}] if system else []) + messages,
                   "temperature": 0.8, "max_tokens": 400},
                  {"Authorization": "Bearer " + os.environ.get("CHAT_API_KEY", "none")}, timeout=60)
        return d["choices"][0]["message"]["content"].strip(), model
    return claude(system or "", messages, FALLBACK_CHAT_MODEL, max_tokens=400).strip(), FALLBACK_CHAT_MODEL


def write(system: str, messages: list[dict]) -> str:
    return claude(system, messages, WRITE_MODEL).strip()
