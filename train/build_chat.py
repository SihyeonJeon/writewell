"""Turn collected threads into chat-format training rows for the small conversational model.

The model's lane (see route/jev.py) is chat: a reaction, a quick answer, a short opinion. So a row is a
real person's post as the user turn and a real reply as the model turn. No text here is generated.

Usage: build_chat.py            # reads data/social/*.jsonl, writes data/train/chat.jsonl + eval.jsonl
Row:   {"messages": [{"role": "user"|"assistant", "content": str}, ...], "source", "board", "year"}
"""
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data"
SRC, OUT = ROOT / "social", ROOT / "train"
OUT.mkdir(parents=True, exist_ok=True)

SKIP_BOARDS = {"sex", "sex_literature", "marvel"}       # explicit / horror fiction: not the chat register we want
MAX_POST, MIN_REPLY, MAX_REPLY, PER_POST = 1200, 6, 300, 3

PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), ""),                                  # email
    (re.compile(r"(?<!\d)09\d{2}[- ]?\d{3}[- ]?\d{3}(?!\d)"), ""),                  # TW mobile
    (re.compile(r"(?i)(line|賴|ig|instagram|fb|微信|wechat)\s*(id)?\s*[:：@＠]\s*[\w.\-]{3,}"), ""),  # handles
    (re.compile(r"@[\w.\-]{3,}"), ""),
]
SPAM = re.compile(r"加賴|加line|借款|貸款|代辦|賺錢機會|兼職.*日領|優惠碼|折扣碼|蝦皮賣場|私訊我|歡迎私訊|抽卡|卡稱|已刪除|此回應已被")
FLOOR = re.compile(r"^\s*((B\d+|b\d+|原po|原PO)[\s,，、:：]*)+")
ONLY_NOISE = re.compile(r"^[\W\d_a-zA-Z]*$")


def clean(s):
    for rx, to in PII:
        s = rx.sub(to, s)
    s = re.sub(r"\n-{2,}\n[^\n]{0,40}$", "", s.strip())            # trailing signature block
    s = re.sub(r"[ \t]+\n", "\n", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def reply_ok(t):
    return MIN_REPLY <= len(t) <= MAX_REPLY and not SPAM.search(t) and not ONLY_NOISE.match(t)


MIN_POST = {"plurk": 8}                                    # a plurk is one line; forum posts need some body
THREAD_SOURCES = {"dcard", "plurk", "bahamut", "mobile01"}     # sources whose records carry a reply thread


def thread_rows(r):
    """One row per good reply: the opening post is the user turn, a real reply is the model turn."""
    src = r["source"]
    if r.get("board") in SKIP_BOARDS or not r.get("comments") or r.get("partial"):
        return
    post = clean(r["text"])
    if len(post) < MIN_POST.get(src, 30) or len(post) > MAX_POST or SPAM.search(post):
        return
    title = re.sub(r"^(RE:)?\s*【[^】]{1,6}】\s*", "", clean(r.get("title", "")))    # Bahamut topic tags
    user = f"{title}\n\n{post}" if title and title not in post[:len(title) + 5] else post
    cands = []
    for c in r["comments"]:
        # the OP's own follow-ups and replies aimed at another commenter need context we do not keep
        if c["op"] or re.search(r"[Bb]\d+|wrote:|引用|樓上|\d+樓", c["text"][:40]):
            continue
        t = clean(c["text"])
        if reply_ok(t):
            cands.append((c.get("likes") or 0, -c["floor"], t))
    year = (r.get("created") or "")[:4]
    for _likes, _f, t in sorted(cands, reverse=True)[:PER_POST]:
        msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": t}]
        yield {"messages": msgs, "source": src, "board": r.get("board", ""),
               "year": year if year.isdigit() else r["ts"][:4]}


def main():
    rows, seen, stats = [], set(), Counter()
    for f in sorted(SRC.glob("*.jsonl")):
        if f.stem not in THREAD_SOURCES:
            continue
        fn = thread_rows
        for line in open(f):
            for row in fn(json.loads(line)) or ():
                h = hashlib.md5(row["messages"][-1]["content"].encode()).hexdigest()
                if h in seen:                                   # the same stock reply under many posts
                    stats["dup_reply"] += 1
                    continue
                seen.add(h)
                rows.append(row)
                stats[row["source"]] += 1
    random.Random(0).shuffle(rows)
    n_eval = min(500, len(rows) // 20)
    for name, part in (("eval", rows[:n_eval]), ("chat", rows[n_eval:])):
        with open(OUT / f"{name}.jsonl", "w") as fo:
            for r in part:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    chars = sum(len(m["content"]) for r in rows for m in r["messages"])
    print(dict(stats), "rows", len(rows), "eval", n_eval, "chars", chars)
    print("years", sorted(Counter(r["year"] for r in rows).items()))
    print("boards", Counter(r["board"] for r in rows).most_common(12))


if __name__ == "__main__":
    main()
