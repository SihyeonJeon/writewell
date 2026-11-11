"""Pre-LLM Taiwanese community / colloquial text at scale, from Common Crawl captures dated <= 2021.

Sibling of cc_human.py (which samples 400 docs per source for n-gram stats). This one has no per-source
cap, covers more crawls, keeps forum replies, and writes to its own directories so both can run at once.

Usage: cc_social.py clusters                # stream-filter cluster.idx for crawls we do not have yet
       cc_social.py index [source ...]      # list every capture under each source prefix (resumable)
       cc_social.py urls  [source ...]      # print URL shape counts per source (to tune the regexes)
       cc_social.py fetch [source ...]      # download WARC records, extract text (resumable)
Output: data/social/<source>.jsonl  (url, ts, crawl, source, genre, title, text)
"""
import gzip
import hashlib
import io
import json
import random
import re
import resource
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import trafilatura
from lxml import etree
from lxml import html as LH
from trafilatura.meta import reset_caches
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cc_human import IDX as HUMAN_IDX, S2T, T2S  # noqa: E402

# Bulk reads go through the commoncrawl S3 bucket with our own AWS credentials (Common Crawl's documented
# path for volume). The anonymous HTTPS endpoint rate-limits hard and answers 403 once you exceed it.
_S3 = boto3.client("s3", region_name="us-east-1",
                   config=Config(max_pool_connections=64, retries={"max_attempts": 8, "mode": "standard"}))


THREADS_N, RSS_LIMIT = 8, 700 * 2**20     # modest on purpose: see the memory guard in fetch()
ERRORS = Counter()          # why reads failed, printed by fetch when a batch goes bad


class _Resp:
    def __init__(self, content):
        self.content = content


def get(url, headers=None):
    key = url.split("data.commoncrawl.org/", 1)[1]
    kw = {"Range": headers["Range"]} if headers and "Range" in headers else {}
    try:
        return _Resp(_S3.get_object(Bucket="commoncrawl", Key=key, **kw)["Body"].read())
    except (ClientError, BotoCoreError) as e:
        ERRORS[type(e).__name__ + ":" + str(e)[:80]] += 1
        return None


def warc_html(d):
    a, n = int(d["offset"]), int(d["length"])
    r = get("https://data.commoncrawl.org/" + d["filename"], headers={"Range": f"bytes={a}-{a + n - 1}"})
    if not r:
        return None
    raw = gzip.GzipFile(fileobj=io.BytesIO(r.content)).read()
    parts = raw.split(b"\r\n\r\n", 2)
    if len(parts) < 3:
        return None
    m = re.search(rb"charset=([\w-]+)", parts[1], re.I) or re.search(rb"charset=[\"']?([\w-]+)", parts[2][:3000], re.I)
    enc = m.group(1).decode() if m else "utf-8"
    try:
        return parts[2].decode(enc, "replace")
    except LookupError:
        return parts[2].decode("utf-8", "replace")

ROOT = Path(__file__).resolve().parent.parent / "data"
IDX, OUT = ROOT / "social_index", ROOT / "social"
IDX.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# Two crawls a year where the index exists; every capture timestamp is before 2022.
CRAWLS = ["CC-MAIN-2021-49", "CC-MAIN-2021-31", "CC-MAIN-2021-10", "CC-MAIN-2020-34", "CC-MAIN-2020-10",
          "CC-MAIN-2019-35", "CC-MAIN-2019-13", "CC-MAIN-2018-34", "CC-MAIN-2018-09", "CC-MAIN-2017-34",
          "CC-MAIN-2017-09", "CC-MAIN-2016-36", "CC-MAIN-2015-48"]

# source: (SURT prefix, article URL regex, genre, max index blocks per crawl, min CJK chars)
SOURCES = {
    "dcard": ("tw,dcard)/f/", r"dcard\.tw/f/[\w-]+/p/\d+", "forum", 400, 80),
    "mobile01": ("com,mobile01)/topicdetail.php", r"topicdetail\.php\?f=\d+&t=\d+", "forum", 400, 80),
    "bahamut": ("tw,com,gamer,forum)/", r"forum\.gamer\.com\.tw/(C|Co)\.php\?", "forum", 400, 80),
    "babyhome": ("tw,com,babyhome,", r"forum\.babyhome\.com\.tw/topic/\d+", "forum", 400, 80),
    "backpackers": ("tw,com,backpackers)/forum/", r"showthread\.php\?(.*&)?t=\d+", "forum", 400, 80),
    "plurk": ("com,plurk)/p/", r"plurk\.com/p/\w+", "microblog", 400, 30),
    "komica": ("org,komica,", r"pixmicat\.php\?res=\d+|read\.php\?key=\d+", "imageboard", 400, 50),
    "ck101": ("com,ck101)/thread", r"ck101\.com/thread-\d+", "forum", 400, 80),
    "gamebase": ("tw,com,gamebase,", r"/forum/\d+/topic/\d+", "forum", 400, 80),
    "xuite": ("net,xuite,blog)/", r"blog\.xuite\.net/[\w.-]+/[\w.-]+/\d+", "personal_blog", 120, 200),
    "pixnet": ("net,pixnet,", r"\.pixnet\.net/blog/post/\d+", "personal_blog", 120, 200),
}


def cluster_lines(crawl):
    """[(key, shard, offset, length)] for the blocks that can hold any of our prefixes."""
    small = IDX / f"{crawl}.cluster.tsv"
    if not small.exists():
        prefixes = [s[0] for s in SOURCES.values()]
        before, hit = {}, []
        full = HUMAN_IDX / f"{crawl}.cluster.idx"
        if full.exists():
            for line in open(full, encoding="utf-8", errors="replace"):
                line = line.rstrip("\n")
                key = line.split(" ", 1)[0]
                keep = False
                for p in prefixes:
                    if key < p:
                        before[p] = line
                    elif key.startswith(p):
                        keep = True
                if keep:
                    hit.append(line)
        else:
            # cluster.idx is ~140MB and sorted: bisect it with Range requests instead of downloading it.
            url = f"https://data.commoncrawl.org/cc-index/collections/{crawl}/indexes/cluster.idx"
            size = _S3.head_object(Bucket="commoncrawl", Key=url.split("data.commoncrawl.org/", 1)[1])["ContentLength"]

            def chunk(a, b):
                return get(url, headers={"Range": f"bytes={max(a, 0)}-{min(b, size) - 1}"}).content

            def lower(target):
                """Byte offset of a line start at or shortly before the first key >= target."""
                lo, hi = 0, size
                while hi - lo > 200_000:
                    mid = (lo + hi) // 2
                    buf = chunk(mid, mid + 8192)
                    nl = buf.find(b"\n")
                    key = buf[nl + 1:].split(b" ", 1)[0].decode("utf-8", "replace")
                    if key < target:
                        lo = mid
                    else:
                        hi = mid
                return lo

            for p in prefixes:
                a, b = lower(p), lower(p + "￿") + 400_000
                buf = chunk(a, b).decode("utf-8", "replace")
                for line in buf.split("\n")[1:-1]:
                    key = line.split(" ", 1)[0]
                    if key < p:
                        before[p] = line
                    elif key.startswith(p):
                        hit.append(line)
        rows = sorted(set(hit) | set(before.values()))
        tmp = small.with_suffix(".tmp")
        tmp.write_text("\n".join(rows) + "\n", encoding="utf-8")
        tmp.rename(small)
        print(crawl, "cluster lines kept", len(rows), flush=True)
    out = []
    for line in open(small, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4:
            out.append((parts[0].split(" ")[0], parts[1], int(parts[2]), int(parts[3])))
    return out


def blocks_for(crawl, prefix):
    rows = cluster_lines(crawl)
    inside = [r for r in rows if r[0].startswith(prefix)]
    lead = [r for r in rows if r[0] < prefix]
    return ([max(lead)] if lead else []) + inside


def index(src):
    prefix, _pat, _genre, cap, _min = SOURCES[src]
    for crawl in CRAWLS:
        out = IDX / f"{src}.{crawl}.jsonl"
        if out.exists() or not (IDX / f"{crawl}.cluster.tsv").exists():  # rerun once `clusters` has caught up
            continue
        blocks = blocks_for(crawl, prefix)
        if len(blocks) > cap:
            blocks = random.Random(src + crawl).sample(blocks, cap)
        rows, seen = [], set()
        for _key, shard, off, ln in blocks:
            r = get(f"https://data.commoncrawl.org/cc-index/collections/{crawl}/indexes/{shard}",
                    headers={"Range": f"bytes={off}-{off + ln - 1}"})
            if not r:
                continue
            for line in gzip.decompress(r.content).decode("utf-8", "replace").splitlines():
                if not line.startswith(prefix):
                    continue
                try:
                    _k, ts, js = line.split(" ", 2)
                    d = json.loads(js)
                except ValueError:
                    continue
                u = d.get("url", "")
                if d.get("status") != "200" or "html" not in d.get("mime-detected", d.get("mime", "html")) or u in seen:
                    continue
                if ts[:4] > "2021":
                    continue
                seen.add(u)
                rows.append({"url": u, "timestamp": ts, "filename": d["filename"], "offset": d["offset"],
                             "length": d["length"], "crawl": crawl})
        tmp = out.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for d in rows:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        tmp.rename(out)
        print(src, crawl, "blocks", len(blocks), "captures", len(rows), flush=True)


def candidates(src):
    rx, seen, rows = re.compile(SOURCES[src][1]), set(), []
    for crawl in CRAWLS:
        f = IDX / f"{src}.{crawl}.jsonl"
        if not f.exists():
            continue
        # a Dcard post also appears as /b/<floor> comment permalinks: take the post page itself when we have it
        for d in sorted(map(json.loads, open(f)), key=lambda d: "/b/" in d["url"]):
            key = canon(src, d["url"])
            if key in seen or not rx.search(d["url"]):
                continue
            seen.add(key)
            rows.append(d)
    return rows


def canon(src, url):
    """One key per article, so the same post captured in several crawls or URL variants is fetched once."""
    u = re.sub(r"^https?://(www\.|m\.)?", "", url).split("#")[0]
    if src == "dcard":
        m = re.search(r"/p/(\d+)", u)
        return m.group(1) if m else u
    if src == "mobile01":
        m = re.search(r"f=(\d+)&t=(\d+)", u)
        return "-".join(m.groups()) + "p" + (re.search(r"[?&]p=(\d+)", u) or [0, "1"])[1] if m else u
    if src in ("pixnet", "xuite", "ck101", "babyhome", "gamebase"):
        return u.split("?")[0]
    if src in ("backpackers", "bahamut"):  # session ids and tracking params vary, the thread id does not
        q = dict(re.findall(r"[?&](t|bsn|snA|sn|page)=(\d+)", u))
        return u.split("?")[0] + "?" + "&".join(f"{k}={v}" for k, v in sorted(q.items()))
    return u


def urls(src):
    shapes = Counter()
    n = 0
    for crawl in CRAWLS:
        f = IDX / f"{src}.{crawl}.jsonl"
        if not f.exists():
            continue
        for line in open(f):
            u = json.loads(line)["url"]
            n += 1
            shapes[re.sub(r"\d+", "N", re.sub(r"^https?://", "", u).split("&")[0])[:70]] += 1
    print(f"== {src}: {n} captures, {len(candidates(src))} unique article candidates")
    for k, v in shapes.most_common(12):
        print(f"   {v:7d}  {k}")


def tw_ok(text, min_cjk):
    cjk = re.findall(r"[一-鿿]", text)
    if len(cjk) < min_cjk:
        return False
    s = "".join(cjk[:3000])
    a, b = S2T.convert(s), T2S.convert(s)
    simp = sum(1 for x, y in zip(s, a) if x != y) if len(a) == len(s) else 0
    trad = sum(1 for x, y in zip(s, b) if x != y) if len(b) == len(s) else 1
    return simp / len(s) <= 0.02 and trad / len(s) >= 0.04


def dcard(html):
    """Post + comment thread from the page's own state JSON. Names, schools and avatars are not kept."""
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if m:  # 2020+ layout
        st = json.loads(m.group(1)).get("props", {}).get("initialState", {})
        pstore, cstore = st.get("post", {}).get("data"), st.get("comment", {}).get("data")
    else:  # 2017-2019 layout
        i = html.find("window.$STATE=")
        if i < 0:
            return None
        st, _ = json.JSONDecoder().raw_decode(html[i + 14:])
        pstore, cstore = st.get("posts", {}).get("store"), st.get("comments", {}).get("store")
    vals = lambda x: list(x.values()) if isinstance(x, dict) else list(x or [])  # noqa: E731 - old layout is a list
    posts = [p for p in vals(pstore) if isinstance(p, dict) and p.get("content")]
    if not posts:
        return None
    p = max(posts, key=lambda x: len(x["content"]))
    clean = lambda s: re.sub(r"https?://\S+", "", s or "").strip()  # noqa: E731 - image/link lines
    cs = [c for c in vals(cstore)
          if isinstance(c, dict) and c.get("content") and not c.get("hidden")
          and str(c.get("postId")) == str(p.get("id"))]
    cs.sort(key=lambda c: c.get("floor", 0))
    return {"title": p.get("title", ""), "text": clean(p["content"]), "created": p.get("createdAt", ""),
            "board": p.get("forumAlias", ""), "likes": p.get("likeCount", 0),
            "comments": [{"floor": c.get("floor"), "op": bool(c.get("host")), "likes": c.get("likeCount", 0),
                          "text": clean(c["content"])} for c in cs if clean(c["content"])]}


def _txt(el, drop=("blockquote", "script", "style", "form", "button")):
    """Visible text of a node with quoted blocks and widgets removed, <br>/<p>/<div> as line breaks."""
    el = LH.fromstring(LH.tostring(el, encoding="unicode"))
    for bad in el.xpath("|".join(f".//{t}" for t in drop)):
        bad.drop_tree()
    for br in el.xpath(".//br|.//p|.//div"):
        br.tail = "\n" + (br.tail or "")
    t = re.sub(r"https?://\S+", "", el.text_content())
    t = re.sub(r"[ \t\u3000\xa0]+", " ", t)
    return re.sub(r"\s*\n\s*", "\n", t).strip()


def plurk(doc):
    main = doc.xpath('//article[contains(@class,"bigplurk")]')
    if not main:
        return None
    main = main[0]
    body = main.xpath('.//*[contains(@class,"plurk_content") or contains(@class,"text_holder")]')
    who = lambda e: "".join(e.xpath('(.//*[contains(@class,"user")]//a)[1]//text()')).strip()  # noqa: E731
    op, when = who(main), "".join(main.xpath('(.//time/@datetime)[1]'))
    comments = []
    for i, li in enumerate(doc.xpath('//ul[contains(@class,"responses")]/li'), 1):
        c = li.xpath('.//*[contains(@class,"plurk_content") or contains(@class,"text_holder")]')
        if c and _txt(c[0]) and not (comments and comments[-1]["text"] == _txt(c[0])):
            comments.append({"floor": i, "op": bool(op) and who(li) == op, "likes": 0, "text": _txt(c[0])})
    return {"text": _txt(body[0]) if body else "", "title": "", "created": when, "comments": comments}


def bahamut(doc):
    posts = doc.xpath('//section[contains(@class,"c-section")][.//div[contains(@class,"c-article__content")]]')
    if not posts:
        return None
    who = lambda e: "".join(e.xpath('(.//a[contains(@class,"userid")])[1]/text()')).strip()  # noqa: E731
    out, op = [], who(posts[0])
    for i, sec in enumerate(posts):
        t = _txt(sec.xpath('.//div[contains(@class,"c-article__content")]')[0])
        if i and t:
            out.append({"floor": i, "op": who(sec) == op, "likes": 0, "text": t})
        for r in sec.xpath('.//div[contains(@class,"c-reply__item")]//article') if i == 0 else []:
            if _txt(r):
                out.append({"floor": 1000 + len(out), "op": False, "likes": 0, "text": _txt(r)})
    first = posts[0]
    return {"text": _txt(first.xpath('.//div[contains(@class,"c-article__content")]')[0]),
            "title": "".join(doc.xpath('(//h1[contains(@class,"c-post__header__title")])[1]//text()')).strip(),
            "created": "".join(first.xpath('(.//a[contains(@class,"edittime")]/@data-mtime)[1]')), "comments": out}


def mobile01(doc):
    first = doc.xpath('//div[@itemprop="articleBody"]')          # only on page 1 of a topic
    post = _txt(first[0]) if first else ""
    replies = [t for t in (_txt(a) for a in doc.xpath('//article[contains(@class,"c-articleLimit")]')) if t and t != post]
    if not post and not replies:
        return None
    return {"text": post, "created": "", "partial": not post,
            "comments": [{"floor": i, "op": False, "likes": 0, "text": t} for i, t in enumerate(replies, 1)],
            "title": re.sub(r"\s*(\(第\d+頁\))?\s*-\s*Mobile01.*$", "", "".join(doc.xpath("//title/text()"))).strip()}


THREADS = {"plurk": plurk, "bahamut": bahamut, "mobile01": mobile01}


def extract(src, html):
    genre = SOURCES[src][2]
    if src in THREADS:
        try:
            d = THREADS[src](LH.fromstring(html))
        except (ValueError, AttributeError, TypeError, IndexError, etree.ParserError):
            d = None
        if d and (d["text"] or d["comments"]):
            return d.pop("text") or d["comments"][0]["text"], d.pop("title"), d
    if src == "dcard":
        try:
            d = dcard(html)
        except (ValueError, AttributeError, TypeError):
            d = None
        if d and d["created"][:4] <= "2021":
            return d.pop("text"), d.pop("title"), d
        if d:
            return None, "", {}
    text = trafilatura.extract(html, include_comments=genre != "personal_blog", include_tables=False,
                               favor_recall=genre != "personal_blog", deduplicate=True)
    meta = trafilatura.extract_metadata(html)
    return text, (meta.title if meta else "") or "", {}


def one(src, d):
    try:
        html = warc_html(d)
        if not html:
            return {"miss": True}       # the read failed: not a verdict on the page, so it must stay retryable
        text, title, extra = extract(src, html)
        whole = (text or "") + "".join(c["text"] for c in extra.get("comments", []))
        if not text or not tw_ok(whole, SOURCES[src][4]):
            return None
        return {"url": d["url"], "ts": d["timestamp"], "crawl": d["crawl"], "source": src,
                "genre": SOURCES[src][2], "title": title, "text": text, **extra}
    except Exception as e:  # noqa: BLE001 - one bad record must not stop the run
        return {"err": repr(e)[:200]}


def head(text):
    return hashlib.md5(re.sub(r"\s", "", text)[:160].encode()).hexdigest()


def fetch(src, limit=None):
    out, tried_f = OUT / f"{src}.jsonl", IDX / f"{src}.tried"
    cands = candidates(src)
    random.Random(src).shuffle(cands)
    have, heads = set(), set()
    if out.exists():
        for x in open(out):
            r = json.loads(x)
            have.add(r["url"])
            heads.add(head(r["text"]))
    tried = set(open(tried_f).read().split("\n")) if tried_f.exists() else set()
    todo = [d for d in cands if d["url"] not in have and d["url"] not in tried]
    if limit:
        todo = todo[:limit]
    n, bad = len(have), 0
    with open(out, "a") as fo, open(tried_f, "a") as ft, ThreadPoolExecutor(THREADS_N) as ex:
        for i in range(0, len(todo), 100):
            batch = todo[i:i + 100]
            results = list(ex.map(lambda d: one(src, d), batch))
            misses = sum(1 for r in results if r and r.get("miss"))
            bad = bad + 1 if misses > len(batch) // 2 else 0
            if misses:
                print(src, "read failures", misses, "/", len(batch), dict(ERRORS.most_common(2)), flush=True)
            if bad >= 5:                 # network or credentials are down: stop instead of burning the queue
                print(src, "stopping: 5 failing batches in a row; rerun to resume", flush=True)
                return
            if bad:
                time.sleep(60)
            for d, r in zip(batch, results):
                if r and r.get("miss"):
                    continue
                ft.write(d["url"] + "\n")
                if not r or "err" in r:
                    continue
                h = head(r["text"])
                if h in heads:
                    continue
                heads.add(h)
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
            fo.flush()
            ft.flush()
            # This laptop runs close to its RAM limit. Parser caches grow over a long run, so drop them each
            # batch and hand back to the shell loop (exit 3 = "restart me") well before we become the problem.
            reset_caches()
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > RSS_LIMIT:
                print(src, "memory guard: restarting to release memory", flush=True)
                sys.exit(3)
            print(src, "kept", n, "tried", i + len(batch), "/", len(todo), flush=True)


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    limit = int(args.pop()) if args and args[-1].isdigit() else None
    srcs = args or list(SOURCES)
    if mode == "clusters":
        with ThreadPoolExecutor(5) as ex:
            list(ex.map(cluster_lines, CRAWLS))
    elif mode == "index":
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(index, srcs))
    elif mode == "urls":
        for s in srcs:
            urls(s)
    else:
        for s in srcs:
            fetch(s, limit)
