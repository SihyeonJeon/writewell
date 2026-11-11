"""Per-turn routing with Jev (TypeSafe's decision model). Jev answers typed questions; this file owns the policy.

Routes: "gemma" (our small model, answers directly) or "opus" with an effort level.
Also returns the output form, which selects the style-gate baseline (chat vs long-form).

Calls the Decisions API through OpenRouter (OPENROUTER_API_KEY) with a pinned Jev version.
Any failure, refusal or low-confidence answer routes to opus: a wrong route should cost money, not quality.
"""
import json
import os
import urllib.request
from dataclasses import dataclass

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.14"          # pinned: the floating alias can change under tuned thresholds
TIMEOUT_S = 3
CONTEXT_TURNS = 3

QUESTIONS = {
    "form": {
        "type": "choice",
        "instructions": "What kind of output does the last user message call for?",
        "criteria": {
            "chat": "A conversational reply: small talk, a quick answer, a reaction, a short opinion.",
            "longform": "A piece of writing to keep or publish: article, post, essay, email, story, speech.",
            "structured": "Code, a table, a list of data, a translation, or another non-prose result.",
        },
    },
    "difficulty": {
        "type": "score",
        "instructions": "How much reasoning does a good answer to the last user message need?",
        "criteria": [
            "None: greeting, chit-chat, or a fact most people know.",
            "A little: a short explanation or a simple rewrite.",
            "Real thought: weighing options, planning a piece of writing, multi-step explanation.",
            "Hard: long analysis, careful argument, technical or mathematical work.",
        ],
    },
    "stakes": {
        "type": "noul",
        "instructions": "Would a wrong or careless answer cause the user real harm or cost "
                        "(health, law, money, safety, or facts they will act on)?",
    },
}


@dataclass
class Route:
    model: str              # "gemma" | "opus"
    effort: str | None      # opus only: "low" | "medium" | "high"
    form: str               # "chat" | "longform" | "structured"
    reason: str
    answers: dict | None = None


FALLBACK = Route("opus", "medium", "longform", "fallback")


def decide(answers: dict) -> Route:
    """Policy over Jev's answers. Pure function, so thresholds can be tuned on logged answers."""
    try:
        form, diff, stakes = answers["form"], answers["difficulty"], answers["stakes"]
        if "refusal" in (form["type"], diff["type"], stakes["type"]):
            return FALLBACK
        kind, d, risky = form["choice"], float(diff["score"]), float(stakes["noul"])
    except (KeyError, TypeError, ValueError):
        return FALLBACK
    sure = form.get("confidence", 0) >= 0.6 and diff.get("confidence", 0) >= 0.5
    if kind == "chat" and d < 0.8 and risky < 0.3 and sure:
        return Route("gemma", None, "chat", f"chat d={d:.2f}", answers)
    effort = "high" if d >= 2.4 or risky >= 0.7 else "medium" if d >= 1.4 or kind == "longform" else "low"
    why = f"{kind} d={d:.2f} stakes={risky:.2f}" + ("" if sure else " unsure")
    return Route("opus", effort, kind if sure else "longform", why, answers)


def route(messages: list[dict]) -> Route:
    """messages: [{'role': 'user'|'assistant', 'content': str}, ...], last one from the user."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return FALLBACK
    state = {"conversation": [{"role": m["role"], "text": m["content"][:4000]} for m in messages[-(2 * CONTEXT_TURNS + 1):]]}
    body = json.dumps({"model": MODEL, "state": state, "questions": QUESTIONS}).encode()
    req = urllib.request.Request(URL, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return decide(json.load(r)["answers"])
    except Exception:  # noqa: BLE001 - timeouts, HTTP errors and shape changes all take the safe route
        return FALLBACK


if __name__ == "__main__":
    import sys
    r = route([{"role": "user", "content": sys.argv[1]}])
    print(r.model, r.effort, r.form, "|", r.reason)
