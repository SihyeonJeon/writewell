"""Pre-LLM Taiwanese human text from Common Crawl captures dated <= 2021.

Proof of date = the crawl timestamp (the page existed in that form on that day). No LLM calls.
Usage: cc_human.py index [source ...]   # list candidate captures per source (resumable)
       cc_human.py fetch [source ...]   # download WARC records, extract text
Output: data/human/<source>.jsonl  (url, ts, source, genre, title, text)
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import trafilatura
import trafilatura.meta
from opencc import OpenCC

ROOT = Path(__file__).resolve().parent.parent / "data"
IDX, HUM = ROOT / "cc_index", ROOT / "human"
IDX.mkdir(parents=True, exist_ok=True)
HUM.mkdir(parents=True, exist_ok=True)
CRAWLS = ["CC-MAIN-2021-31", "CC-MAIN-2020-34", "CC-MAIN-2019-35", "CC-MAIN-2018-34"]
UA = {"User-Agent": "corpus-research/0.1"}
PER_SOURCE = 400
T2S, S2T = OpenCC("t2s"), OpenCC("s2t")

# source: (domain query, article URL regex, genre)
SOURCES = {
    "pixnet": ("pixnet.net", r"\.pixnet\.net/blog/post/\d+", "personal_blog"),
    "ptt": ("ptt.cc", r"ptt\.cc/bbs/[\w-]+/M\.\d+\.A\.\w+\.html", "forum"),
    "mobile01": ("mobile01.com", r"mobile01\.com/topicdetail\.php\?f=\d+&t=\d+$", "forum"),
    "dcard": ("dcard.tw", r"dcard\.tw/f/\w+/p/\d+", "forum"),
    "yahoo_answers": ("tw.answers.yahoo.com", r"question/index\?qid=", "qa"),
    "cna": ("cna.com.tw", r"cna\.com\.tw/news/\w+/\d{12}\.aspx", "news"),
    "ltn": ("ltn.com.tw", r"ltn\.com\.tw/(news|article)/\w+/(breakingnews|paper)/\d+", "news"),
    "udn": ("udn.com", r"udn\.com/news/story/\d+/\d+", "news"),
    "thenewslens": ("thenewslens.com", r"thenewslens\.com/article/\d+", "feature"),
    "twreporter": ("twreporter.org", r"twreporter\.org/a/[\w-]+$", "feature"),
    "cw": ("cw.com.tw", r"cw\.com\.tw/article/\d+", "magazine"),
    "commonhealth": ("commonhealth.com.tw", r"commonhealth\.com\.tw/article/\d+", "magazine"),
    "gvm": ("gvm.com.tw", r"gvm\.com\.tw/article/\d+", "magazine"),
    "managertoday": ("managertoday.com.tw", r"managertoday\.com\.tw/articles/view/\d+", "magazine"),
    "womany": ("womany.net", r"womany\.net/read/article/\d+", "lifestyle"),
    "icook": ("icook.tw", r"icook\.tw/recipes/\d+$", "howto"),
    "ithome": ("ithome.com.tw", r"ithome\.com\.tw/(news|tech|article)/\d+", "tech"),
    "inside": ("inside.com.tw", r"inside\.com\.tw/article/\d+", "tech"),
    "techbang": ("techbang.com", r"techbang\.com/posts/\d+", "tech"),
    "heho": ("heho.com.tw", r"heho\.com\.tw/archives/\d+", "health"),
    "vocus": ("vocus.cc", r"vocus\.cc/(article|[\w@-]+)/[0-9a-f]{24}", "essay"),
    "matters": ("matters.news", r"matters\.news/@[\w-]+/", "essay"),
}


def get(url, **kw):
    for i in range(6):
        try:
            r = requests.get(url, headers=UA | kw.pop("headers", {}), timeout=90, **kw)
            if r.status_code in (200, 206):
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(3 * (i + 1) + random.random() * 3)
    return None


SURT = {
    "pixnet": "net,pixnet,", "ptt": "cc,ptt)/bbs/", "mobile01": "com,mobile01)/topicdetail.php",
    "dcard": "tw,dcard)/f/", "yahoo_answers": "com,yahoo,answers,tw)/question", "cna": "tw,com,cna)/news/",
    "ltn": "tw,com,ltn,news)/", "udn": "com,udn)/news/story/", "thenewslens": "com,thenewslens)/article/",
    "twreporter": "org,twreporter)/a/", "cw": "tw,com,cw)/article/", "commonhealth": "tw,com,commonhealth)/article/",
    "gvm": "tw,com,gvm)/article/", "managertoday": "tw,com,managertoday)/articles/view/",
    "womany": "net,womany)/read/article/", "icook": "tw,icook)/recipes/", "ithome": "tw,com,ithome)/",
    "inside": "tw,com,inside)/article/", "techbang": "com,techbang)/posts/", "heho": "tw,com,heho)/archives/",
    "vocus": "cc,vocus)/", "matters": "news,matters)/@",
}
_CLUSTER = {}


def cluster(crawl):
    """cluster.idx: one line per ~3000 index records -> (first key, shard file, offset, length)."""
    if crawl not in _CLUSTER:
        f = IDX / f"{crawl}.cluster.idx"
        if not f.exists():
            r = get(f"https://data.commoncrawl.org/cc-index/collections/{crawl}/indexes/cluster.idx")
            f.write_bytes(r.content)
        keys, rows = [], []
        for line in open(f, encoding="utf-8", errors="replace"):
            k, shard, off, ln, _ = line.rstrip("\n").split("\t")
            keys.append(k.split(" ")[0])
            rows.append((shard, int(off), int(ln)))
        _CLUSTER[crawl] = (keys, rows)
    return _CLUSTER[crawl]


def index(src):
    import bisect
    pat = SOURCES[src][1]
    prefix = SURT[src]
    out, done = IDX / f"{src}.jsonl", IDX / f"{src}.done"
    if done.exists():
        return
    rx, seen, rows = re.compile(pat), set(), []
    for crawl in CRAWLS:
        keys, blocks = cluster(crawl)
        lo = max(bisect.bisect_left(keys, prefix) - 1, 0)
        hi = bisect.bisect_right(keys, prefix + "\uffff")
        order = list(range(lo, hi))
        random.Random(src + crawl).shuffle(order)
        for n, i in enumerate(order[:40]):
            shard, off, ln = blocks[i]
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
                if d.get("status") != "200" or "html" not in d.get("mime-detected", "html") or not rx.search(u) or u in seen:
                    continue
                seen.add(u)
                rows.append({"url": u, "timestamp": ts, "filename": d["filename"], "offset": d["offset"],
                             "length": d["length"], "crawl": crawl})
            if len(rows) >= PER_SOURCE * 6:
                break
        print(src, crawl, "blocks", hi - lo, "candidates", len(rows), flush=True)
        with open(out, "w") as f:
            for d in rows:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        if len(rows) >= PER_SOURCE * 6:
            break
    done.touch()


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


def ptt_text(html):
    m = re.search(r'<div id="main-content"[^>]*>(.*?)(?:<span class="f2">※ 發信站|<div class="push">|$)', html, re.S)
    if not m:
        return None
    body = re.sub(r'<div class="article-metaline(-right)?">.*?</div>\s*</div>|<div class="article-metaline(-right)?">.*?</div>', "", m.group(1), flags=re.S)
    body = re.sub(r"<[^>]+>", "", body)
    body = body.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").replace("&quot;", '"')
    lines = [x for x in body.split("\n") if not re.match(r"\s*(:|※|>|作者|標題|時間|看板)", x)]
    text = "\n".join(lines).split("\n--\n")[0]
    # PTT hard-wraps lines at ~40 columns: join lines inside a paragraph
    paras = [re.sub(r"\s*\n\s*", "", p) for p in re.split(r"\n\s*\n", text)]
    return "\n".join(p for p in paras if p.strip())


def script_ok(text):
    cjk = re.findall(r"[一-鿿]", text)
    if len(cjk) < 300:
        return False
    s = "".join(cjk[:3000])
    simp_only = sum(1 for a, b in zip(s, S2T.convert(s)) if a != b) if len(S2T.convert(s)) == len(s) else 0
    trad_only = sum(1 for a, b in zip(s, T2S.convert(s)) if a != b) if len(T2S.convert(s)) == len(s) else 1
    return simp_only / len(s) <= 0.02 and trad_only / len(s) >= 0.05


def one(src, d):
    try:
        if src == "ptt":
            ts = int(re.search(r"/M\.(\d+)\.", d["url"]).group(1))
            if ts > 1640995200:
                return None
        html = warc_html(d)
        if not html:
            return None
        if src == "ptt":
            text = ptt_text(html)
            t = re.search(r'<meta property="og:title" content="([^"]*)"', html)
            title = t.group(1) if t else ""
        else:
            text = trafilatura.extract(html, include_comments=False, include_tables=False, favor_precision=True)
            meta = trafilatura.extract_metadata(html)
            title = (meta.title if meta else "") or ""
        if not text or not script_ok(text):
            return None
        return {"url": d["url"], "ts": d["timestamp"], "crawl": d["crawl"], "source": src,
                "genre": SOURCES[src][2], "title": title, "text": text}
    except Exception as e:  # noqa: BLE001 - one bad record must not stop the run
        return {"err": repr(e)[:200]}


def fetch(src):
    f_idx, out = IDX / f"{src}.jsonl", HUM / f"{src}.jsonl"
    if not f_idx.exists():
        return
    cands = [json.loads(x) for x in open(f_idx)]
    random.Random(src).shuffle(cands)
    have, heads = set(), set()
    if out.exists():
        for x in open(out):
            r = json.loads(x)
            have.add(r["url"])
            heads.add(hashlib.md5(re.sub(r"\s", "", r["text"])[:120].encode()).hexdigest())
    tried_f = IDX / f"{src}.tried"
    tried = set(open(tried_f).read().split("\n")) if tried_f.exists() else set()
    todo = [d for d in cands if d["url"] not in have and d["url"] not in tried]
    n = len(have)
    with open(out, "a") as fo, open(tried_f, "a") as ft, ThreadPoolExecutor(12) as ex:
        for i in range(0, len(todo), 48):
            if n >= PER_SOURCE:
                break
            batch = todo[i:i + 48]
            for d, r in zip(batch, ex.map(lambda d: one(src, d), batch)):
                ft.write(d["url"] + "\n")
                if not r or "err" in r:
                    continue
                h = hashlib.md5(re.sub(r"\s", "", r["text"])[:120].encode()).hexdigest()
                if h in heads:
                    continue
                heads.add(h)
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
            fo.flush()
            ft.flush()
            trafilatura.meta.reset_caches()      # trafilatura/lxml caches grow without bound otherwise
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
            print(src, "kept", n, "tried", i + len(batch), "/", len(todo), f"rss {rss:.2f}GB", flush=True)
            if rss > 1.5:
                raise SystemExit(f"{src}: memory guard at {rss:.1f} GB - rerun to resume")


if __name__ == "__main__":
    mode, srcs = sys.argv[1], sys.argv[2:] or list(SOURCES)
    fn = index if mode == "index" else fetch
    if mode == "index":
        for c in CRAWLS:
            cluster(c)
        with ThreadPoolExecutor(6) as ex:
            list(ex.map(fn, srcs))
    else:
        for s in srcs:
            fn(s)
