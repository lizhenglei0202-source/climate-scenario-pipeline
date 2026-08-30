import argparse
import os
import sys
import asyncio
import textwrap
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Iterable, List, Dict, Any

import requests

# Requires browser-use: pip install browser-use langchain-openai
# Configure OPENAI_API_KEY (or MOONSHOT_API_KEY) and OPENAI_BASE_URL in .env

# Where harvested PDFs are written. Override with --output-dir or the
# HARVEST_DOWNLOAD_DIR environment variable.
DEFAULT_DOWNLOAD_DIR = os.environ.get("HARVEST_DOWNLOAD_DIR", "./pdf_corpus")
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
PLAYWRIGHT_PDF_KEYWORDS = [
    "PDF",
    "Download PDF",
    "Full Text PDF",
    "View PDF",
    "Get PDF",
    "Article PDF",
    "全文PDF",
    "下载 PDF",
    "下载PDF",
    "全文下载",
]
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0 Safari/537.36 "
        "ScholarAgent/1.0"
    )
}


def load_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE lines from a .env file into os.environ."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if (val.startswith('"') and val.endswith('"')) or (
                    val.startswith("'") and val.endswith("'")
                ):
                    val = val[1:-1]
                if key:
                    os.environ[key] = val
    except FileNotFoundError:
        pass


def build_browser(profile_directory: str, download_dir: Optional[str] = None):
    """Configure connection to the existing Chrome profile on macOS."""
    from browser_use import Browser


    return Browser(
        executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        user_data_dir=os.path.expanduser('~/Library/Application Support/Google/Chrome'),
        profile_directory=profile_directory,
        headless=False,
        accept_downloads=True,
        downloads_path=download_dir if download_dir else os.path.expanduser('~/Downloads'),
    )


def build_task_for_titles(titles: List[str], download_dir: str, max_pages: int = 50) -> str:
    """为指定的论文标题列表构建 WoS 下载任务。"""
    wos_url = "https://webofscience.clarivate.cn/wos/alldb/basic-search"


    move_file_code = textwrap.dedent(
        f"""
        import glob
        import os
        import shutil
        import time

        search_dir = os.path.expanduser('~/Downloads')
        target_dir = r"{download_dir}"
        os.makedirs(target_dir, exist_ok=True)

        def latest_pdf(path: str):
            files = [f for f in glob.glob(os.path.join(path, '*.pdf')) if os.path.isfile(f)]
            if not files:
                return None
            files.sort(key=os.path.getmtime)
            return files[-1]

        deadline = time.time() + 180
        previous_size = None
        while time.time() < deadline:
            if glob.glob(os.path.join(search_dir, '*.crdownload')):
                time.sleep(1)
                previous_size = None
                continue

            candidate = latest_pdf(search_dir)
            if candidate is None:
                time.sleep(1)
                continue

            size = os.path.getsize(candidate)
            if previous_size == size:
                destination = os.path.join(target_dir, os.path.basename(candidate))
                if os.path.exists(destination):
                    base, ext = os.path.splitext(destination)
                    index = 1
                    while os.path.exists(f"{{base}}_{{index}}{{ext}}"):
                        index += 1
                    destination = f"{{base}}_{{index}}{{ext}}"
                shutil.move(candidate, destination)
                print(f"Moved {{candidate}} to {{destination}}")
                break

            previous_size = size
            time.sleep(1)
        else:
            print('No finished PDF was detected in ~/Downloads during the allowed time window.')
        """
    ).strip()

    titles_list = "\n".join(f"{i+1}. {title}" for i, title in enumerate(titles))

    return (
        f"## TASK: Download specific papers from Web of Science\n\n"
        f"**GOAL**: Download the following {len(titles)} papers using WoS full-text access\n\n"

        f"### PAPER LIST TO DOWNLOAD:\n"
        f"{titles_list}\n\n"

        f"### INSTRUCTIONS:\n"
        f"For EACH paper in the list above:\n\n"

        f"**Step 1 - SEARCH:**\n"
        f"1. Navigate to {wos_url}\n"
        f"2. Search for the paper title (use quotes for exact match)\n"
        f"3. If no results, try searching with key phrases from the title\n\n"

        f"**Step 2 - OPEN RECORD:**\n"
        f"1. Click on the matching paper title\n"
        f"2. Verify it's the correct paper by checking authors/year if available\n\n"

        f"**Step 3 - DOWNLOAD PDF:**\n"
        f"1. Look for 'Full Text from Publisher' or similar links\n"
        f"2. Click to go to publisher's website\n"
        f"3. Find and click the PDF download button\n"
        f"4. Labels to look for: 'PDF', 'Download PDF', 'View PDF', '下载PDF'\n\n"

        f"**Step 4 - MOVE FILE:**\n"
        f"Run this Python code to move the downloaded PDF:\n"
        f"```python\n{move_file_code}\n```\n\n"

        f"**Step 5 - CONTINUE:**\n"
        f"1. Close extra tabs\n"
        f"2. Go back to WoS\n"
        f"3. Process the next paper in the list\n\n"

        f"### IMPORTANT:\n"
        f"- Process papers in order (1, 2, 3, ...)\n"
        f"- If a paper cannot be found or downloaded, skip it and continue with the next\n"
        f"- Stop when all papers have been attempted\n"
    )


def build_task(query: str, max_pdfs: int, download_dir: str, max_pages: int = 50) -> str:
    wos_url = "https://webofscience.clarivate.cn/wos/alldb/smart-search"


    move_file_code = textwrap.dedent(
        f"""
        import glob
        import os
        import shutil
        import time

        search_dir = os.path.expanduser('~/Downloads')
        target_dir = r"{download_dir}"
        os.makedirs(target_dir, exist_ok=True)

        def latest_pdf(path: str):
            files = [f for f in glob.glob(os.path.join(path, '*.pdf')) if os.path.isfile(f)]
            if not files:
                return None
            files.sort(key=os.path.getmtime)
            return files[-1]

        deadline = time.time() + 180
        previous_size = None
        while time.time() < deadline:
            if glob.glob(os.path.join(search_dir, '*.crdownload')):
                time.sleep(1)
                previous_size = None
                continue

            candidate = latest_pdf(search_dir)
            if candidate is None:
                time.sleep(1)
                continue

            size = os.path.getsize(candidate)
            if previous_size == size:
                destination = os.path.join(target_dir, os.path.basename(candidate))
                if os.path.exists(destination):
                    base, ext = os.path.splitext(destination)
                    index = 1
                    while os.path.exists(f"{{base}}_{{index}}{{ext}}"):
                        index += 1
                    destination = f"{{base}}_{{index}}{{ext}}"
                shutil.move(candidate, destination)
                print(f"Moved {{candidate}} to {{destination}}")
                break

            previous_size = size
            time.sleep(1)
        else:
            print('No finished PDF was detected in ~/Downloads during the allowed time window.')
        """
    ).strip()

    count_downloads_code = textwrap.dedent(
        f"""
        import glob
        import os
        target_dir = r"{download_dir}"
        pdf_files = glob.glob(os.path.join(target_dir, '*.pdf'))
        pdf_count = len(pdf_files)
        print(f"CURRENT_DOWNLOAD_COUNT: {{pdf_count}}")
        if pdf_files:
            print("DOWNLOADED_FILES:")
            for f in sorted(pdf_files)[-10:]:
                print(f"  - {{os.path.basename(f)}}")
        """
    ).strip()

    processed_titles_file = os.path.join(download_dir, ".processed_titles.txt")

    check_title_code = textwrap.dedent(
        f"""
        import os
        processed_file = r"{processed_titles_file}"
        def is_title_processed(title):
            if not os.path.exists(processed_file):
                return False
            with open(processed_file, 'r', encoding='utf-8') as f:
                processed = set(line.strip().lower() for line in f)
            return title.strip().lower() in processed

        def mark_title_processed(title):
            os.makedirs(os.path.dirname(processed_file), exist_ok=True)
            with open(processed_file, 'a', encoding='utf-8') as f:
                f.write(title.strip() + '\\n')
            print(f"Marked as processed: {{title[:50]}}...")


        # mark_title_processed("<paper title>")
        """
    ).strip()

    return (
        f"## TASK: Download PDFs from Web of Science\n\n"
        f"**GOAL**: Download {max_pdfs} PDFs to `{download_dir}`\n\n"

        f"### SETUP: Title Tracking Functions\n"
        f"First, define these helper functions for tracking processed titles:\n"
        f"```python\n{check_title_code}\n```\n\n"

        f"### PHASE 1: SEARCH\n"
        f"1. Navigate to {wos_url}\n"
        f"2. Enter search query: `{query}`\n"
        f"3. Click the Search button\n"
        f"4. Wait for results to load\n\n"

        f"### PHASE 2: PROCESS EACH RESULT (ONE AT A TIME)\n\n"
        f"**CRITICAL RULES:**\n"
        f"- Process results **sequentially by their visible row number** (1, 2, 3, ...)\n"
        f"- **NEVER skip back to row 1** - always process the next unprocessed number\n"
        f"- **Track your current row number** - after processing row N, next is N+1\n"
        f"- Use the title tracking to avoid re-downloading the same paper\n\n"

        f"**For EACH result (starting from row 1):**\n\n"
        f"**Step A - CHECK PROGRESS:**\n"
        f"Run this Python code to check download count:\n"
        f"```python\n{count_downloads_code}\n```\n"
        f"If CURRENT_DOWNLOAD_COUNT >= {max_pdfs}, STOP immediately.\n\n"

        f"**Step B - LOCATE THE ROW:**\n"
        f"- Find the result row with the current target number\n"
        f"- Row numbers are displayed on the LEFT side of each result\n"
        f"- If target row not on current page, go to PHASE 3 (pagination)\n\n"

        f"**Step C - CHECK IF ALREADY PROCESSED:**\n"
        f"- Read the paper title from the row\n"
        f"- Run: `if is_title_processed(\"<paper title>\"): print(\"SKIP\")`\n"
        f"- If SKIP, increment row number and go to Step A\n\n"

        f"**Step D - OPEN RECORD:**\n"
        f"- Click on the **title link** to open the record detail page\n\n"

        f"**Step E - FIND FULL TEXT:**\n"
        f"- Look for these links (in order):\n"
        f"  1. 'Full Text from Publisher' / 'Publisher Full Text'\n"
        f"  2. 'Free Full Text'\n"
        f"  3. 'View Full Text'\n"
        f"  4. Any link containing 'PDF'\n"
        f"- Click to go to publisher's website\n\n"

        f"**Step F - DOWNLOAD PDF:**\n"
        f"- Find and click the PDF download button\n"
        f"- Labels: 'PDF', 'Download PDF', 'View PDF', '下载PDF'\n\n"

        f"**Step G - MOVE FILE AND MARK PROCESSED:**\n"
        f"Run this to move the downloaded PDF:\n"
        f"```python\n{move_file_code}\n```\n"
        f"Then mark the title as processed:\n"
        f"```python\nmark_title_processed(\"<paper title>\")\n```\n\n"

        f"**Step H - CLEANUP AND CONTINUE:**\n"
        f"- Close ALL extra tabs\n"
        f"- Return to WoS search results page\n"
        f"- **INCREMENT row number by 1** (NEVER go back to 1!)\n"
        f"- Go to Step A\n\n"

        f"### PHASE 3: PAGINATION\n\n"
        f"**When to paginate:**\n"
        f"- When target row number > max visible row on current page\n"
        f"- Typically after processing rows 1-50 (or 1-10/1-25 depending on settings)\n\n"

        f"**How to paginate:**\n"
        f"1. Find pagination controls at bottom of results\n"
        f"2. Click 'Next' / '下一页' / '>' button\n"
        f"3. Wait for new page to load\n"
        f"4. **Continue with your current target row number**\n"
        f"   - If you finished row 50, next target is 51\n"
        f"   - New page shows rows 51-100, etc.\n\n"

        f"**Limits:**\n"
        f"- Max pages: {max_pages}\n"
        f"- If no 'Next' button, task is complete\n\n"

        f"### PHASE 4: STOPPING CONDITIONS\n"
        f"Stop when ANY of these occur:\n"
        f"1. Downloaded PDFs >= {max_pdfs}\n"
        f"2. No more results available\n"
        f"3. Processed {max_pages} pages\n"
        f"4. No 'Next Page' button\n\n"

        f"### CRITICAL REMINDERS\n"
        f"❗ **ALWAYS increment row number after each attempt**\n"
        f"❗ **NEVER go back to row 1 during the same session**\n"
        f"❗ **If download fails, still increment and continue**\n"
        f"❗ **Row numbers are continuous across pages** (1-50, 51-100, ...)\n"
        f"❗ **Use title tracking to avoid duplicates**\n"
    )


def safe_filename(name: str, fallback: str = "document") -> str:
    sanitized = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip())
    sanitized = sanitized.strip("_")
    if not sanitized:
        sanitized = fallback
    return sanitized[:120]


def ensure_unique_path(base_dir: Path, filename: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    candidate = base_dir / filename
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    counter = 1
    while True:
        candidate = base_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def resolve_chrome_profile_path(base_dir: Optional[str], profile: str) -> Path:
    root = Path(base_dir or "~/Library/Application Support/Google/Chrome").expanduser()
    if profile:
        return root / profile
    return root


def create_session(extra_headers: Optional[Dict[str, str]] = None) -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    if extra_headers:
        session.headers.update(extra_headers)
    return session


def download_pdf_from_url(
    url: str,
    download_dir: Path,
    filename_hint: str,
    session: Optional[requests.Session] = None,
    timeout: int = 90,
) -> Optional[Path]:
    session = session or create_session()
    try:
        response = session.get(url, stream=True, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[下载失败] {url}: {exc}")
        return None

    content_type = (response.headers.get("Content-Type") or "").lower()
    iterator = response.iter_content(chunk_size=8192)
    first_chunk = b""
    for chunk in iterator:
        if chunk:
            first_chunk = chunk
            break
    if not first_chunk:
        print(f"[空响应] 未从 {url} 读取到任何内容")
        return None

    if "pdf" not in content_type and not first_chunk.startswith(b"%PDF"):
        print(f"[非PDF] {url} 的 Content-Type={content_type}")
        return None

    filename = safe_filename(filename_hint) + ".pdf"
    destination = ensure_unique_path(download_dir, filename)
    try:
        with open(destination, "wb") as f:
            f.write(first_chunk)
            for chunk in iterator:
                if chunk:
                    f.write(chunk)
        print(f"[已保存] {destination.name} <- {url}")
        return destination
    except OSError as exc:
        print(f"[写入失败] {destination}: {exc}")
        if destination.exists():
            destination.unlink(missing_ok=True)
        return None


def chunk_iterable(items: Iterable[str], size: int) -> Iterable[List[str]]:
    chunk: List[str] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def search_pubmed_pmids(
    query: str,
    max_results: int,
    session: requests.Session,
    api_key: Optional[str] = None,
    email: Optional[str] = None,
) -> List[str]:
    pmids: List[str] = []
    retstart = 0
    batch_size = 200
    while len(pmids) < max_results:
        retmax = min(batch_size, max_results - len(pmids))
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retstart": retstart,
            "retmax": retmax,
        }
        if api_key:
            params["api_key"] = api_key
        if email:
            params["email"] = email
        try:
            response = session.get(PUBMED_SEARCH_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            print(f"[PubMed搜索错误] {exc}")
            break

        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            break
        pmids.extend(id_list)
        retstart += len(id_list)
        total = int(data.get("esearchresult", {}).get("count", retstart))
        if retstart >= total:
            break
    return pmids[:max_results]


def parse_pubmed_xml(xml_text: str) -> List[Dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        print(f"[PubMed解析错误] {exc}")
        return []

    articles: List[Dict[str, Any]] = []
    for article in root.findall("PubmedArticle"):
        pmid = article.findtext(".//PMID")
        title = article.findtext(".//ArticleTitle") or ""
        doi = ""
        for article_id in article.findall(".//ArticleIdList/ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = (article_id.text or "").strip()
                break
        journal = article.findtext(".//Journal/Title") or ""
        articles.append(
            {
                "pmid": pmid or "",
                "title": title.strip(),
                "doi": doi,
                "journal": journal.strip(),
            }
        )
    return articles


def fetch_pubmed_articles(
    pmids: List[str],
    session: requests.Session,
    api_key: Optional[str] = None,
    email: Optional[str] = None,
) -> Iterable[Dict[str, Any]]:
    for batch in chunk_iterable(pmids, 50):
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
        }
        if api_key:
            params["api_key"] = api_key
        if email:
            params["email"] = email
        try:
            response = session.get(PUBMED_FETCH_URL, params=params, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"[PubMed详情错误] {exc}")
            continue
        for record in parse_pubmed_xml(response.text):
            yield record


def run_pubmed_pipeline(
    query: str,
    max_pdfs: int,
    download_dir: Path,
    email: Optional[str] = None,
    api_key: Optional[str] = None,
) -> int:
    if max_pdfs <= 0:
        return 0
    session = create_session()
    pmids = search_pubmed_pmids(query, max_pdfs * 2, session, api_key=api_key, email=email)
    if not pmids:
        print("[PubMed] 未检索到 PMID，跳过。")
        return 0

    downloaded = 0
    seen = set()
    for article in fetch_pubmed_articles(pmids, session, api_key=api_key, email=email):
        pmid = article.get("pmid")
        if not pmid or pmid in seen:
            continue
        seen.add(pmid)
        title = article.get("title") or f"pubmed_{pmid}"
        filename_hint = f"pubmed_{pmid}_{title}"

        pdf_candidates = [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/pdf/"]
        doi = article.get("doi")
        if doi:
            pdf_candidates.append(f"https://doi.org/{doi}")

        for pdf_url in pdf_candidates:
            saved_path = download_pdf_from_url(pdf_url, download_dir, filename_hint, session=session)
            if saved_path:
                downloaded += 1
                break
        if downloaded >= max_pdfs:
            break
    print(f"[PubMed] 完成下载 {downloaded} 篇。")
    return downloaded


def query_semantic_scholar(
    query: str,
    limit: int,
    session: requests.Session,
    fields: str = "title,openAccessPdf,url,year,paperId,doi",
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    offset = 0
    page_size = 100
    while len(records) < limit:
        params = {
            "query": query,
            "limit": min(page_size, limit - len(records)),
            "offset": offset,
            "fields": fields,
        }
        try:
            response = session.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            print(f"[Semantic Scholar错误] {exc}")
            break

        papers = payload.get("data", [])
        if not papers:
            break
        records.extend(papers)
        offset += len(papers)
        total = payload.get("total")
        if total is not None and offset >= total:
            break
    return records[:limit]


def run_semantic_scholar_pipeline(
    query: str,
    max_pdfs: int,
    download_dir: Path,
    api_key: Optional[str] = None,
) -> int:
    if max_pdfs <= 0:
        return 0
    session_headers = {}
    if api_key:
        session_headers["x-api-key"] = api_key
    session = create_session(session_headers)

    papers = query_semantic_scholar(
        query,
        limit=max_pdfs * 2,
        session=session,
        fields="title,openAccessPdf,url,year,paperId",
    )
    downloaded = 0
    for paper in papers:
        pdf_info = paper.get("openAccessPdf") or {}
        pdf_url = pdf_info.get("url")
        if not pdf_url:
            continue
        title = paper.get("title") or paper.get("paperId") or "semantic_scholar"
        filename_hint = f"semantic_{title}"
        saved_path = download_pdf_from_url(pdf_url, download_dir, filename_hint, session=session)
        if saved_path:
            downloaded += 1
        if downloaded >= max_pdfs:
            break
    print(f"[Semantic Scholar] 完成下载 {downloaded} 篇。")
    return downloaded


def parse_arxiv_feed(xml_text: str) -> List[Dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        print(f"[arXiv解析错误] {exc}")
        return []

    ns = "{http://www.w3.org/2005/Atom}"
    entries = []
    for entry in root.findall(f"{ns}entry"):
        title = (entry.findtext(f"{ns}title") or "").strip()
        arxiv_id = entry.findtext(f"{ns}id") or ""
        pdf_url = None
        for link in entry.findall(f"{ns}link"):
            href = link.get("href")
            rel = link.get("rel")
            link_type = link.get("type")
            title_attr = link.get("title")
            if not href:
                continue
            if link_type == "application/pdf" or title_attr == "pdf" or (rel == "related" and href.endswith(".pdf")):
                pdf_url = href
                break
        if not pdf_url and arxiv_id:
            if "/abs/" in arxiv_id:
                pdf_url = arxiv_id.replace("/abs/", "/pdf/")
            else:
                pdf_url = arxiv_id
            if not pdf_url.endswith(".pdf"):
                pdf_url += ".pdf"
        entries.append(
            {
                "title": title or arxiv_id,
                "pdf_url": pdf_url,
                "id": arxiv_id,
            }
        )
    return entries


def run_arxiv_pipeline(
    query: str,
    max_pdfs: int,
    download_dir: Path,
) -> int:
    if max_pdfs <= 0:
        return 0
    session = create_session()
    downloaded = 0
    start = 0
    page_size = 100

    normalized_query = query.strip()
    if not normalized_query:
        normalized_query = "all"
    first_token = normalized_query.split()[0]
    if ":" in first_token:
        search_query = normalized_query
    else:
        search_query = f"all:{normalized_query}"

    while downloaded < max_pdfs:
        max_results = min(page_size, max_pdfs - downloaded)
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            response = session.get(ARXIV_API_URL, params=params, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"[arXiv请求错误] {exc}")
            break

        entries = parse_arxiv_feed(response.text)
        if not entries:
            break

        for entry in entries:
            pdf_url = entry.get("pdf_url")
            if not pdf_url:
                continue
            title = entry.get("title") or entry.get("id") or "arxiv"
            filename_hint = f"arxiv_{title}"
            saved_path = download_pdf_from_url(pdf_url, download_dir, filename_hint, session=session)
            if saved_path:
                downloaded += 1
            if downloaded >= max_pdfs:
                break

        start += len(entries)
        if len(entries) < max_results:
            break

    print(f"[arXiv] 完成下载 {downloaded} 篇。")
    return downloaded


async def run_semantic_scholar_browser_pipeline(
    query: str,
    max_pdfs: int,
    download_dir: Path,
    user_data_dir: Optional[str],
    profile: Optional[str],
    api_key: Optional[str] = None,
    browser_channel: str = "chrome",
) -> int:
    if max_pdfs <= 0:
        return 0
    try:
        from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
    except ImportError:
        print("[Semantic Scholar 浏览下载] Playwright 未安装。请运行 `pip install playwright` 并执行 `playwright install chrome`。")
        return 0

    session_headers = {}
    if api_key:
        session_headers["x-api-key"] = api_key
    session = create_session(session_headers)
    papers = query_semantic_scholar(
        query,
        limit=max_pdfs * 2,
        session=session,
        fields="title,url,openAccessPdf,doi",
    )
    if not papers:
        print("[Semantic Scholar 浏览下载] 未检索到足够的记录。")
        return 0

    resolved_profile = resolve_chrome_profile_path(user_data_dir, profile or "")
    resolved_profile.mkdir(parents=True, exist_ok=True)
    download_count = 0

    async def save_download(download, title_hint: str) -> Optional[Path]:
        suggested = download.suggested_filename or "document.pdf"
        suffix = Path(suggested).suffix or ".pdf"
        filename = safe_filename(title_hint, "s2_browser") + suffix
        destination = ensure_unique_path(download_dir, filename)
        await download.save_as(str(destination))
        print(f"[Semantic Scholar 浏览下载] 已保存 {destination.name}")
        return destination

    async def click_keywords(page) -> Optional[Any]:
        for keyword in PLAYWRIGHT_PDF_KEYWORDS:
            locator = page.get_by_text(keyword, exact=False)
            if await locator.count():
                try:
                    async with page.expect_download(timeout=60000) as download_info:
                        await locator.first.click()
                    return await download_info.value
                except PlaywrightTimeoutError:
                    continue
        anchors = page.locator("a[href$='.pdf']")
        if await anchors.count():
            try:
                async with page.expect_download(timeout=60000) as download_info:
                    await anchors.first.click()
                return await download_info.value
            except PlaywrightTimeoutError:
                return None
        return None

    async def handle_single_target(page, target_url: str, title: str) -> bool:
        nonlocal download_count
        try:
            if target_url.lower().endswith(".pdf"):
                try:
                    async with page.expect_download(timeout=60000) as download_info:
                        await page.goto(target_url, wait_until="load", timeout=60000)
                    download = await download_info.value
                    await save_download(download, title)
                    return True
                except PlaywrightTimeoutError:
                    pass

            await page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
            download = await click_keywords(page)
            if download:
                await save_download(download, title)
                return True
        except PlaywrightTimeoutError:
            print(f"[Semantic Scholar 浏览下载] 访问 {target_url} 超时。")
        except Exception as exc:
            print(f"[Semantic Scholar 浏览下载] 处理 {target_url} 失败: {exc}")
        return False

    async with async_playwright() as playwright:
        context = await playwright.chromium.launch_persistent_context(
            user_data_dir=str(resolved_profile),
            channel=browser_channel,
            headless=False,
            accept_downloads=True,
        )
        page = context.pages[0] if context.pages else await context.new_page()

        for paper in papers:
            title = paper.get("title") or paper.get("paperId") or "semantic_scholar"
            candidates = []
            pdf_info = paper.get("openAccessPdf") or {}
            pdf_url = pdf_info.get("url")
            if pdf_url:
                candidates.append(pdf_url)
            doi = paper.get("doi")
            if doi:
                candidates.append(f"https://doi.org/{doi}")
            canonical_url = paper.get("url")
            if canonical_url:
                candidates.append(canonical_url)
            if not candidates:
                continue

            success = False
            for url in candidates:
                if await handle_single_target(page, url, title):
                    success = True
                    download_count += 1
                    break
            if success:
                print(f"[Semantic Scholar 浏览下载] 当前累计 {download_count}/{max_pdfs}")
            if download_count >= max_pdfs:
                break
        await context.close()

    print(f"[Semantic Scholar 浏览下载] 完成下载 {download_count} 篇。")
    return download_count


def configure_llm_env(model: str) -> None:
    """Configure environment variables for the chosen LLM provider."""
    if model.startswith("kimi"):
        moonshot_key = os.environ.get("MOONSHOT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not moonshot_key:
            print("ERROR: MOONSHOT_API_KEY not set.", file=sys.stderr)
        else:
            os.environ["OPENAI_API_KEY"] = moonshot_key
            base = os.environ.get("MOONSHOT_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.moonshot.cn/v1"
            os.environ["OPENAI_BASE_URL"] = base
    elif model.startswith("gpt-"):
        if os.environ.get("OPENAI_API_KEY") is None:
            print(f"WARNING: OPENAI_API_KEY is not set explicitly for model '{model}'.", file=sys.stderr)


async def run_wos_agent(
    query: str,
    max_pdfs: int,
    download_dir: Path,
    profile: str,
    model: str,
    max_pages: int,
) -> None:
    print(f"Initializing Browser (Profile: {profile})...")
    browser = build_browser(profile, download_dir=str(download_dir))

    from browser_use import Agent, ChatOpenAI

    task_prompt = build_task(query, max_pdfs, str(download_dir), max_pages=max_pages)

    agent = Agent(
        task=task_prompt,
        browser=browser,
        llm=ChatOpenAI(model=model),
    )

    print("Agent started working...")
    await agent.run()

    print("-" * 40)
    print("✅ Task Completed.")
    print(f"📂 PDFs should be saved in: {download_dir}")
    print("-" * 40)


async def run_wos_agent_for_titles(
    titles: List[str],
    download_dir: Path,
    profile: str,
    model: str,
    use_playwright: bool = True,
) -> int:
    """使用 WoS 下载指定标题的论文列表。"""
    if not titles:
        print("[WoS 补充下载] 没有需要下载的论文。")
        return 0

    if use_playwright:

        from wos_playwright import run_wos_playwright_download
        results = await run_wos_playwright_download(titles, download_dir, profile)
        success_count = sum(1 for r in results if r.success)
        return success_count
    else:

        print(f"[WoS 补充下载] 准备下载 {len(titles)} 篇非开源论文...")
        print(f"Initializing Browser (Profile: {profile})...")
        browser = build_browser(profile, download_dir=str(download_dir))

        from browser_use import Agent, ChatOpenAI

        task_prompt = build_task_for_titles(titles, str(download_dir))

        agent = Agent(
            task=task_prompt,
            browser=browser,
            llm=ChatOpenAI(model=model),
        )

        print("Agent started working on non-open-access papers...")
        await agent.run()

        print("-" * 40)
        print("✅ WoS supplementary download completed.")
        print(f"📂 PDFs should be saved in: {download_dir}")
        print("-" * 40)
        return len(titles)


def run_two_stage_pipeline(
    query: str,
    max_papers: int,
    download_dir: Path,
    api_key: Optional[str] = None,
) -> tuple[int, List[Dict[str, Any]]]:
    """
    两阶段下载流程：
    1. 用 Semantic Scholar 检索论文
    2. 下载所有开源的（有 openAccessPdf）
    3. 返回已下载数量和未下载的论文列表（供 WoS 补充）

    返回: (已下载数量, 未下载的论文列表)
    """
    session_headers = {}
    if api_key:
        session_headers["x-api-key"] = api_key
    session = create_session(session_headers)

    print(f"[两阶段流程] 正在用 Semantic Scholar 检索: {query}")
    papers = query_semantic_scholar(
        query,
        limit=max_papers * 2,
        session=session,
        fields="title,openAccessPdf,url,year,paperId,doi",
    )

    if not papers:
        print("[两阶段流程] Semantic Scholar 未检索到结果。")
        return 0, []

    print(f"[两阶段流程] 检索到 {len(papers)} 篇论文")

    downloaded = 0
    non_open_access: List[Dict[str, Any]] = []

    for paper in papers:
        if downloaded >= max_papers:
            break

        title = paper.get("title") or paper.get("paperId") or "unknown"
        pdf_info = paper.get("openAccessPdf") or {}
        pdf_url = pdf_info.get("url")

        if pdf_url:
            filename_hint = f"s2_{title}"
            saved_path = download_pdf_from_url(pdf_url, download_dir, filename_hint, session=session)
            if saved_path:
                downloaded += 1
                print(f"[开源] 已下载: {title[:60]}...")
            else:
                non_open_access.append(paper)
        else:
            non_open_access.append(paper)

    print(f"\n[两阶段流程] 开源下载完成: {downloaded} 篇")
    print(f"[两阶段流程] 待 WoS 补充下载: {len(non_open_access)} 篇")

    return downloaded, non_open_access


async def run_two_stage_with_wos(
    query: str,
    max_papers: int,
    download_dir: Path,
    profile: str,
    model: str,
    api_key: Optional[str] = None,
    wos_batch_size: int = 20,
    use_playwright: bool = True,
) -> Dict[str, int]:
    """
    完整的两阶段下载流程：
    1. Semantic Scholar 检索 + 开源下载
    2. WoS 补充下载非开源论文

    参数:
        wos_batch_size: 每批发送给 WoS agent 的论文数量（防止任务过长）
        use_playwright: True 使用纯 Playwright（推荐），False 使用 browser_use + LLM
    """

    open_access_count, non_open_access = run_two_stage_pipeline(
        query, max_papers, download_dir, api_key
    )

    wos_success_count = 0
    wos_attempted = 0
    if non_open_access:
        remaining_needed = max_papers - open_access_count
        papers_for_wos = non_open_access[:remaining_needed]

        if papers_for_wos:
            titles = [p.get("title", "") for p in papers_for_wos if p.get("title")]
            wos_attempted = len(titles)


            for i in range(0, len(titles), wos_batch_size):
                batch = titles[i:i + wos_batch_size]
                print(f"\n[WoS] 处理第 {i//wos_batch_size + 1} 批，共 {len(batch)} 篇...")
                batch_success = await run_wos_agent_for_titles(
                    batch, download_dir, profile, model, use_playwright=use_playwright
                )
                wos_success_count += batch_success

    return {
        "open_access": open_access_count,
        "wos_success": wos_success_count,
        "wos_attempted": wos_attempted,
        "total_success": open_access_count + wos_success_count,
    }


async def run_all(args: argparse.Namespace) -> None:
    configure_llm_env(args.model)
    final_dir = Path(args.download_dir).expanduser()
    final_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, str] = {}


    if args.two_stage_max > 0:
        use_playwright = not args.use_llm
        print("\n" + "=" * 50)
        print("🚀 启用两阶段下载模式")
        print("  阶段1: Semantic Scholar 检索 + 开源 PDF 直接下载")
        print(f"  阶段2: WoS 补充下载（{'Playwright 自动化' if use_playwright else 'browser_use + LLM'}）")
        print("=" * 50 + "\n")

        results = await run_two_stage_with_wos(
            args.query,
            args.two_stage_max,
            final_dir,
            args.profile,
            args.model,
            api_key=args.semantic_api_key,
            wos_batch_size=args.wos_batch_size,
            use_playwright=use_playwright,
        )
        summary["开源下载"] = f"{results['open_access']} 篇成功"
        summary["WoS补充"] = f"{results['wos_success']}/{results['wos_attempted']} 篇成功"
        summary["总计"] = f"{results['total_success']} 篇"
        print("[两阶段下载] 已完成。")

        if summary:
            print("\n📊 下载摘要：")
            for source, description in summary.items():
                print(f" - {source}: {description}")
        return

    if args.pubmed_max > 0:
        count = run_pubmed_pipeline(
            args.query,
            args.pubmed_max,
            final_dir,
            email=args.pubmed_email,
            api_key=args.pubmed_api_key,
        )
        summary["PubMed"] = f"{count} 篇"
        print("[PubMed] 阶段已完成。")
    else:
        print("[PubMed] 已跳过。")

    if args.semantic_max > 0:
        count = run_semantic_scholar_pipeline(
            args.query,
            args.semantic_max,
            final_dir,
            api_key=args.semantic_api_key,
        )
        summary["Semantic Scholar"] = f"{count} 篇"
        print("[Semantic Scholar] 阶段已完成。")
    else:
        print("[Semantic Scholar] 已跳过。")

    if args.arxiv_max > 0:
        count = run_arxiv_pipeline(
            args.query,
            args.arxiv_max,
            final_dir,
        )
        summary["arXiv"] = f"{count} 篇"
        print("[arXiv] 阶段已完成。")
    else:
        print("[arXiv] 已跳过。")

    if args.s2_browser_max > 0:
        count = await run_semantic_scholar_browser_pipeline(
            args.query,
            args.s2_browser_max,
            final_dir,
            user_data_dir=args.s2_browser_user_data_dir,
            profile=args.s2_browser_profile or args.profile,
            api_key=args.semantic_api_key,
            browser_channel=args.s2_browser_channel,
        )
        summary["Semantic Scholar（Chrome自动下载）"] = f"{count} 篇"
        print("[Semantic Scholar 浏览下载] 阶段已完成。")
    else:
        print("[Semantic Scholar 浏览下载] 已跳过。")

    if not args.skip_wos and args.max_pdfs > 0:
        await run_wos_agent(
            args.query,
            args.max_pdfs,
            final_dir,
            args.profile,
            args.model,
            args.wos_max_pages,
        )
        summary["Web of Science"] = f"目标 {args.max_pdfs} 篇（browser_use 自动执行）"
        print("[Web of Science] 阶段已完成。")
    else:
        print("[Web of Science] 已跳过。")

    if summary:
        print("\n📊 下载摘要：")
        for source, description in summary.items():
            print(f" - {source}: {description}")
    else:
        print("\n⚠️ 未选择任何下载通道。")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="同时利用 Web of Science、PubMed、Semantic Scholar、arXiv 检索与下载文献。"
    )
    parser.add_argument("query", help="检索关键词或语句。")
    parser.add_argument(
        "max_pdfs",
        nargs="?",
        type=int,
        default=500,
        help="Web of Science 目标下载数量（通过 browser_use 自动化）。",
    )
    parser.add_argument(
        "download_dir",
        nargs="?",
        default=DEFAULT_DOWNLOAD_DIR,
        help=f"PDF 存储目录（默认: {DEFAULT_DOWNLOAD_DIR}）。",
    )
    parser.add_argument(
        "profile",
        nargs="?",
        default="Default",
        help="Chrome Profile 名称，用于 browser_use。",
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=os.environ.get("MODEL", "gemini-1.5-flash"),
        help="LLM 模型名称。",
    )
    parser.add_argument(
        "--pubmed-max",
        type=int,
        default=0,
        help="PubMed API 下载的目标数量（默认 0 为关闭）。",
    )
    parser.add_argument(
        "--semantic-max",
        type=int,
        default=0,
        help="Semantic Scholar API 下载的目标数量（默认 0 为关闭）。",
    )
    parser.add_argument(
        "--arxiv-max",
        type=int,
        default=0,
        help="arXiv API 下载的目标数量（默认 0 为关闭）。",
    )
    parser.add_argument(
        "--s2-browser-max",
        type=int,
        default=0,
        help="Semantic Scholar API 仅取 DOI/URL，再由 Chrome 自动下载的目标数量（默认 0 为关闭）。",
    )
    parser.add_argument(
        "--s2-browser-user-data-dir",
        default=None,
        help="Chrome User Data 根目录（默认为 ~/Library/Application Support/Google/Chrome）。",
    )
    parser.add_argument(
        "--s2-browser-profile",
        default=None,
        help="供 Semantic Scholar 浏览下载使用的 Chrome Profile 名称（默认与主 profile 相同）。",
    )
    parser.add_argument(
        "--s2-browser-channel",
        default="chrome",
        help="Playwright 启动的 Chrome 渠道（默认 chrome，可选 chromium）。",
    )
    parser.add_argument(
        "--skip-wos",
        action="store_true",
        help="跳过 Web of Science browser_use 自动化。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印任务提示，不真正执行。",
    )
    parser.add_argument(
        "--wos-max-pages",
        type=int,
        default=10,
        help="Web of Science 最大翻页次数。",
    )
    parser.add_argument(
        "--pubmed-email",
        default=os.environ.get("PUBMED_EMAIL"),
        help="PubMed API email 标识（可选但推荐）。",
    )
    parser.add_argument(
        "--pubmed-api-key",
        default=os.environ.get("NCBI_API_KEY") or os.environ.get("PUBMED_API_KEY"),
        help="NCBI PubMed API Key（可选）。",
    )
    parser.add_argument(
        "--semantic-api-key",
        default=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"),
        help="Semantic Scholar API Key（可选）。",
    )
    parser.add_argument(
        "--two-stage-max",
        type=int,
        default=0,
        help="两阶段模式目标数量：先用 Semantic Scholar 检索，开源的直接下载，非开源的用 WoS 补充。",
    )
    parser.add_argument(
        "--wos-batch-size",
        type=int,
        default=20,
        help="WoS 补充下载时每批处理的论文数量（默认 20）。",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="WoS 下载使用 browser_use + LLM（默认使用纯 Playwright 自动化）。",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    load_env_file(".env")
    args = parse_args(sys.argv[1:])
    expanded_dir = str(Path(args.download_dir).expanduser())

    print("--- 多通道 Scholar Agent ---")
    print(f"Query: {args.query}")
    print(f"Download Directory: {expanded_dir}")
    print(f"LLM Model: {args.model}")
    print(f"Chrome Profile: {args.profile}")
    if args.two_stage_max > 0:
        wos_method = "browser_use + LLM" if args.use_llm else "Playwright 自动化"
        print(f"🚀 两阶段模式: {args.two_stage_max} 篇")
        print(f"   阶段1: Semantic Scholar 检索 + 开源下载")
        print(f"   阶段2: WoS 补充下载 ({wos_method})")
        print(f"   WoS 批次大小: {args.wos_batch_size}")
    else:
        print(f"Web of Science Target: {args.max_pdfs} (skip={args.skip_wos})")
        print(f"PubMed Target: {args.pubmed_max}")
        print(f"Semantic Scholar Target: {args.semantic_max}")
        print(f"arXiv Target: {args.arxiv_max}")
        print(f"S2 Chrome Download Target: {args.s2_browser_max}")
    if (not args.skip_wos and args.max_pdfs > 0) or args.s2_browser_max > 0 or args.two_stage_max > 0:
        print("\nIMPORTANT PRE-FLIGHT CHECK:")
        if not args.skip_wos and args.max_pdfs > 0:
            print("  - 为 Web of Science 自动化：确保所有 Chrome 窗口关闭，并提前登录 WOS。")
        if args.s2_browser_max > 0:
            print("  - 为 Semantic Scholar 浏览下载：Playwright 将启动 Chrome 持久化上下文，请确保对应 Profile 未被其它窗口占用。")

    if args.dry_run:
        print("\nDRY RUN: 浏览器任务提示如下：")
        if not args.skip_wos and args.max_pdfs > 0:
            print(build_task(args.query, args.max_pdfs, expanded_dir, max_pages=args.wos_max_pages))
        if args.pubmed_max > 0:
            print(f"\n[PubMed] 将通过 API 计划下载 {args.pubmed_max} 篇。")
        if args.semantic_max > 0:
            print(f"\n[Semantic Scholar] 将通过 API 计划下载 {args.semantic_max} 篇。")
        if args.arxiv_max > 0:
            print(f"\n[arXiv] 将通过 API 计划下载 {args.arxiv_max} 篇。")
        if args.s2_browser_max > 0:
            print(f"\n[Semantic Scholar 浏览下载] 将调用 Chrome 自动化下载 {args.s2_browser_max} 篇。")
        sys.exit(0)

    try:
        asyncio.run(run_all(args))
    except KeyboardInterrupt:
        print("\nUser stopped the script.")
    finally:
        print("\n🔴 PROGRAM FINISHED. 浏览器窗口会保持打开状态。")
        print(f"   PLEASE CHECK FILES AT: {expanded_dir}")
        input("   Press [Enter] to close the script (this may close the browser connection)...")
