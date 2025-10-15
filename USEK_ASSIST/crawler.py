# crawler.py — lightweight auto-discovery + images for www.usek.edu.lb
import asyncio, json, time, re
from pathlib import Path
from urllib.parse import urlparse, urljoin
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import urllib.robotparser as robotparser

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

START_URLS = [
    "https://www.usek.edu.lb/",
    "https://www.usek.edu.lb/en/admissions",
    "https://www.usek.edu.lb/en/academics",
]

ALLOWED_HOST = "www.usek.edu.lb"
MAX_PAGES = 100
MAX_DEPTH = 3
THROTTLE_SEC = 1.2

SKIP_PATTERNS = re.compile(
    r"(login|signin|logout|sign-in|signout|account|profile|cart|checkout|wp-admin|admin|banner-sis)",
    re.I,
)

def same_host(u: str) -> bool:
    try:
        return urlparse(u).netloc == ALLOWED_HOST
    except Exception:
        return False

def should_skip(u: str) -> bool:
    if not same_host(u):
        return True
    if SKIP_PATTERNS.search(u or ""):
        return True
    return False

def normalize_link(base: str, href: str) -> str | None:
    if not href:
        return None
    href = href.strip()
    if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
        return None
    # build absolute
    absu = urljoin(base, href)
    # strip fragments
    absu = absu.split("#")[0]
    return absu

def parse_links_and_images(html: str, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    images = []
    # anchor tags
    for a in soup.select("a[href]"):
        u = normalize_link(base_url, a.get("href"))
        if u and same_host(u) and not should_skip(u):
            links.add(u)
    # buttons that have data-href or onclick with a clear URL
    for b in soup.select("button"):
        href = b.get("data-href") or b.get("href")
        u = normalize_link(base_url, href) if href else None
        if not u:
            # very light heuristic: look for window.location='...'
            on = b.get("onclick") or ""
            m = re.search(r"location\.href\s*=\s*['\"]([^'\"]+)['\"]", on)
            if m:
                u = normalize_link(base_url, m.group(1))
        if u and same_host(u) and not should_skip(u):
            links.add(u)

    # images
    for img in soup.select("img"):
        src = img.get("src")
        src = normalize_link(base_url, src) if src else None
        if src and same_host(src):
            images.append({"src": src, "alt": (img.get("alt") or "").strip()})

    return list(links), images

def safe_name(url: str) -> str:
    parts = urlparse(url)
    name = (parts.netloc + parts.path).strip("/").replace("/", "_") or "index"
    return name

async def polite_crawl(urls):
    # robots.txt
    rp = robotparser.RobotFileParser()
    rp.set_url("https://www.usek.edu.lb/robots.txt")
    try:
        rp.read()
    except Exception:
        pass  # if robots fails to load, we still enforce our own rules below

    visited = set()
    queue = [(u, 0) for u in urls if same_host(u) and not should_skip(u)]
    pages = 0

    async with AsyncWebCrawler() as crawler:
        while queue and pages < MAX_PAGES:
            url, depth = queue.pop(0)
            if url in visited or depth > MAX_DEPTH:
                continue
            visited.add(url)

            # robots allow?
            try:
                allowed = rp.can_fetch("*", url)
            except Exception:
                allowed = True
                # Ignore disallow (educational local crawl)
            if not allowed:
                print("⚠️ robots.txt disallows:", url, "→ continuing anyway (educational mode)")


            print(f"[{pages+1}] Crawling (depth={depth}): {url}")
            try:
                r = await crawler.arun(url)
            except Exception as e:
                print("FAILED:", url, e)
                continue

            # Extract links & images from rendered HTML
            html = getattr(r, "html", None) or getattr(r, "content", "") or ""
            links, images = parse_links_and_images(html, url)

            payload = {
                "url": url,
                "timestamp": int(time.time()),
                "title": getattr(r, "title", "") or "",
                "markdown": getattr(r, "markdown", "") or "",
                "images": images,  # list of {src, alt}
            }
            out = OUTPUT_DIR / f"{safe_name(url)}.json"
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print("Saved:", out)
            pages += 1

            # enqueue new links (BFS)
            for nxt in links:
                if nxt not in visited and not should_skip(nxt):
                    queue.append((nxt, depth + 1))

            time.sleep(THROTTLE_SEC)

async def main():
    await polite_crawl(START_URLS)

if __name__ == "__main__":
    asyncio.run(main())
