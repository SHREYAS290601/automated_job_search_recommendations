import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent
LINKS_FILE = PROJECT_ROOT / "links" / "links.txt"
DATA_DIR = PROJECT_ROOT / "data"
STATE_FILE = DATA_DIR / "seen_jobs.json"
RESULTS_FILE = DATA_DIR / "job_results.json"

MAX_JOBS_PER_RUN = 5000  # Ingest all new jobs from all links per run (no per-run cap in practice)


API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

client: Optional[OpenAI] = None
if API_KEY:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL or None)


@dataclass
class JobEntry:
    id: str
    title: str
    url: str
    source_repo: str
    raw_line: str
    added_at: str  # ISO timestamp
    resume_suggestion: Optional[str] = None
    is_faang: bool = False
    age_days: Optional[int] = None  # Speedyapply: e.g. 0, 1, 5 (newest first when sorting asc)
    date_posted: Optional[str] = None  # Jobright: e.g. "Mar 11" (for display and sort)
    location: Optional[str] = None  # Location from table (e.g. "Indianapolis, IN", "New York")
    salary: Optional[str] = None  # Speedyapply FAANG table only (e.g. "$172k/yr")


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_seen_ids() -> set:
    if not STATE_FILE.is_file():
        return set()
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return set(data.get("seen_ids", []))
    except Exception:
        return set()


def _save_seen_ids(ids: set) -> None:
    _ensure_data_dir()
    STATE_FILE.write_text(json.dumps({"seen_ids": sorted(ids)}, indent=2), encoding="utf-8")


def _load_existing_results() -> List[JobEntry]:
    if not RESULTS_FILE.is_file():
        return []
    try:
        data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        out = []
        for item in data.get("jobs", []):
            item = dict(item)
            item.setdefault("is_faang", False)
            item.setdefault("age_days", None)
            item.setdefault("date_posted", None)
            item.setdefault("location", None)
            item.setdefault("salary", None)
            j = JobEntry(**item)
            # Backfill location/age/date_posted/salary from raw_line for old entries that don't have them
            _backfill_job_from_raw_line(j)
            out.append(j)
        return out
    except Exception:
        return []


def _backfill_job_from_raw_line(j: JobEntry) -> None:
    """Populate location, age_days, date_posted, salary from raw_line when missing (old data)."""
    raw = (j.raw_line or "").strip()
    if not raw:
        return
    parts = [p.strip() for p in raw.split("|")]
    # Jobright: | Company | Job Title | Location | Work Model | Date Posted |
    if "**[" in raw and raw.count("|") >= 5:
        if j.location is None and len(parts) > 3:
            loc = parts[3]
            if loc and not loc.startswith("http") and "<" not in loc and len(loc) < 200:
                j.location = loc
        if j.date_posted is None and len(parts) > 5:
            dp = parts[5]
            if dp and not dp.startswith("http") and len(dp) <= 15:
                j.date_posted = dp
    # Speedyapply: <a>...</a> | Position | Location | [Salary] | <a>...</a> | Age
    if "<strong>" in raw and "<a href=" in raw:
        if j.location is None and len(parts) > 3:
            loc = parts[3]
            if loc and "<" not in loc and ">" not in loc and not re.match(r"^\$[\d,]+k?/yr", loc) and len(loc) < 200:
                j.location = loc
        if j.age_days is None and len(parts) >= 5:
            age_cell = (parts[-1] or "").strip()
            m = re.match(r"^(\d+)d\s*$", age_cell, re.I)
            if m:
                j.age_days = int(m.group(1))
        # Salary: FAANG table has $172k/yr in column index 4
        if j.salary is None and len(parts) > 4:
            sal = parts[4].strip()
            if sal and re.match(r"^\$[\d,]+k?/yr", sal):
                j.salary = sal


def _save_results(jobs: List[JobEntry]) -> None:
    _ensure_data_dir()
    RESULTS_FILE.write_text(
        json.dumps({"jobs": [asdict(j) for j in jobs]}, indent=2),
        encoding="utf-8",
    )


def _github_url_to_raw(url: str) -> Optional[str]:
    """
    Convert a GitHub URL to raw content URL.
    - blob URL: .../org/repo/blob/BRANCH/path → raw.../org/repo/BRANCH/path
    - repo root: .../org/repo → raw.../org/repo/HEAD/README.md
    """
    url = url.strip()
    # Blob style: https://github.com/org/repo/blob/master/README.md or .../main/NEW_GRAD_USA.md
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$", url)
    if m:
        org, repo, branch, path = m.group(1), m.group(2), m.group(3), m.group(4)
        return f"https://raw.githubusercontent.com/{org}/{repo}/{branch}/{path}"
    # Repo root: default to README on HEAD
    m = re.match(r"https://github\.com/([^/]+)/([^/#]+)", url)
    if m:
        org, repo = m.group(1), m.group(2)
        return f"https://raw.githubusercontent.com/{org}/{repo}/HEAD/README.md"
    return None


def _fetch_markdown(url: str, timeout: int = 10) -> Optional[str]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; job-scanner/1.0)"}
    try:
        resp = requests.get(url, timeout=timeout, headers=headers)
        if resp.status_code == 200:
            return resp.text
        # If blob URL used "master" and got 404, try "main" (common default branch)
        if resp.status_code == 404 and "/master/" in url:
            fallback = url.replace("/master/", "/main/", 1)
            resp2 = requests.get(fallback, timeout=timeout, headers=headers)
            if resp2.status_code == 200:
                return resp2.text
    except Exception:
        pass
    return None


# --- Jobright-style: markdown table | Company | Job Title | Location | Work Model | Date Posted |
# Company and Job Title are **[\w](url)**; the Job Title URL is the actual job posting.
_JOBRIGHT_LINK_RE = re.compile(r"\*\*\[([^\]]*)\]\((https?://[^\s)]+)\)\*\*")


def _parse_jobright_table(markdown: str, source_repo: str) -> List[JobEntry]:
    """Parse jobright-ai style README: table with Company | Job Title | Location | Work Model | Date Posted."""
    jobs: List[JobEntry] = []
    now = datetime.utcnow().isoformat()
    for line in markdown.splitlines():
        if "|" not in line or "**[" not in line:
            continue
        # Expect at least two **[text](url)** in the line (Company, Job Title)
        links = _JOBRIGHT_LINK_RE.findall(line)
        if len(links) < 2:
            continue
        company_name = (links[0][0] or "").strip()
        job_title = (links[1][0] or "").strip()
        job_url = (links[1][1] or "").strip()
        if not job_url:
            continue
        # Skip non–job-posting links (e.g. "explore jobright.ai" or footer links)
        if "/jobs/" not in job_url and "jobright.ai" in job_url and "/info/" not in job_url:
            continue
        if (job_title or "").strip().lower() == "jobright.ai" or (company_name or "").strip().lower() == "jobright.ai":
            continue
        title = f"{company_name} - {job_title}" if company_name and job_title else (job_title or company_name or "Job")
        job_id = f"{source_repo}|{title}|{job_url}"
        pipe_parts = [p.strip() for p in line.split("|")]
        # Columns: 0=empty, 1=Company, 2=Job Title, 3=Location, 4=Work Model, 5=Date Posted
        date_posted: Optional[str] = None
        location: Optional[str] = None
        if len(pipe_parts) > 5:
            date_cell = pipe_parts[5].strip()
            if date_cell and not date_cell.startswith("http") and len(date_cell) <= 15:
                date_posted = date_cell
        if len(pipe_parts) > 3:
            loc_cell = pipe_parts[3].strip()
            if loc_cell and not loc_cell.startswith("http") and "<" not in loc_cell and len(loc_cell) < 200:
                location = loc_cell
        jobs.append(
            JobEntry(
                id=job_id,
                title=title,
                url=job_url,
                source_repo=source_repo,
                raw_line=line.strip(),
                added_at=now,
                is_faang=False,
                date_posted=date_posted,
                location=location,
            )
        )
    return jobs


# --- Speedyapply-style: HTML tables with TABLE_START (Other) and TABLE_FAANG_START
# Row: <a href="company_url"><strong>Company</strong></a> | Position | Location | [Salary] | <a href="posting_url">...</a> | Age
_SPEEDYAPPLY_COMPANY_RE = re.compile(r'<a\s+href="(https?://[^"]+)"[^>]*>\s*<strong>([^<]+)</strong>\s*</a>', re.I)
_SPEEDYAPPLY_POSTING_RE = re.compile(r'<a\s+href="(https?://[^"]+)"[^>]*>\s*<img', re.I)


def _parse_speedyapply_tables(markdown: str, source_repo: str) -> List[JobEntry]:
    """Parse speedyapply 2026-AI-College-Jobs style: TABLE_START (Other) and TABLE_FAANG_START (FAANG) HTML tables."""
    jobs: List[JobEntry] = []
    now = datetime.utcnow().isoformat()
    in_faang_section = False
    for line in markdown.splitlines():
        if "<!-- TABLE_FAANG_START -->" in line:
            in_faang_section = True
            continue
        if "<!-- TABLE_FAANG_END -->" in line:
            in_faang_section = False
            continue
        if "<!-- TABLE_START -->" in line or "<!-- TABLE_END -->" in line:
            continue
        if "<strong>" not in line or "<a href=" not in line:
            continue
        company_m = _SPEEDYAPPLY_COMPANY_RE.search(line)
        posting_m = _SPEEDYAPPLY_POSTING_RE.search(line)
        if not company_m or not posting_m:
            continue
        company_url, company_name = company_m.group(1), company_m.group(2).strip()
        posting_url = posting_m.group(1).strip()
        pipe_parts = [p.strip() for p in line.split("|")]
        if len(pipe_parts) < 4:
            continue
        position = ""
        for i, col in enumerate(pipe_parts):
            if i < 2:
                continue
            if col and "<" not in col and ">" not in col and len(col) > 1:
                position = col
                break
        # Location is 3rd column (index 3) in both Other and FAANG tables
        location: Optional[str] = None
        if len(pipe_parts) > 3:
            loc_cell = pipe_parts[3].strip()
            if loc_cell and "<" not in loc_cell and ">" not in loc_cell and len(loc_cell) < 200:
                location = loc_cell
        title = f"{company_name} - {position}" if position else company_name
        job_id = f"{source_repo}|{title}|{posting_url}"
        # Age is last column: e.g. 0d, 1d, 52d
        age_days_val: Optional[int] = None
        if len(pipe_parts) >= 5:
            age_cell = (pipe_parts[-1] or "").strip()
            age_match = re.match(r"^(\d+)d\s*$", age_cell, re.I)
            if age_match:
                age_days_val = int(age_match.group(1))
        # Salary: FAANG table has column index 4 = Salary (e.g. $172k/yr)
        salary_val: Optional[str] = None
        if in_faang_section and len(pipe_parts) > 5:
            sal_cell = pipe_parts[4].strip()
            if sal_cell and re.match(r"^\$[\d,]+k?/yr", sal_cell):
                salary_val = sal_cell
        jobs.append(
            JobEntry(
                id=job_id,
                title=title,
                url=posting_url,
                source_repo=source_repo,
                raw_line=line.strip(),
                added_at=now,
                is_faang=in_faang_section,
                age_days=age_days_val,
                location=location,
                salary=salary_val,
            )
        )
    return jobs


def _detect_format_and_parse(markdown: str, source_repo: str) -> List[JobEntry]:
    """Detect job list format and dispatch to the right parser."""
    if "TABLE_FAANG_START" in markdown or "TABLE_START" in markdown:
        return _parse_speedyapply_tables(markdown, source_repo)
    if "| Company |" in markdown and ("Job Title" in markdown or "Position" in markdown):
        return _parse_jobright_table(markdown, source_repo)
    return []


def _read_resume_text() -> str:
    """
    Read templates/resume.tex if available and roughly strip LaTeX to plain text.
    Used for generating resume suggestions.
    """
    resume_path = PROJECT_ROOT / "templates" / "resume.tex"
    if not resume_path.is_file():
        return ""
    text = resume_path.read_text(encoding="utf-8", errors="ignore")
    # Very rough LaTeX -> plain text
    text = re.sub(r"\\[a-zA-Z]+\\*?\\{([^{}]*)\\}", r"\\1", text)
    text = re.sub(r"\\[a-zA-Z]+(\\[[^\\]]*\\])?(\\{[^{}]*\\})?", "", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\\%", "%", text)
    text = re.sub(r"\\&", "&", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def _generate_resume_suggestion(job: JobEntry, resume_text: str, model_name: str) -> Optional[str]:
    """
    Legacy hook for LLM suggestions (now disabled to avoid hallucinated per-job bullets).

    We keep the function for backwards compatibility / tests, but it simply returns None.
    """
    return None


def run_job_scan(model_name: str = "gpt-oss-120b") -> Tuple[int, int]:
    """
    Scan all repos in links/links.txt, find new entry-level jobs (non-intern),
    optionally generate one resume suggestion each, and persist results.

    Returns: (new_jobs_count, total_jobs_in_results)
    """
    _ensure_data_dir()
    seen_ids = _load_seen_ids()
    existing = _load_existing_results()
    existing_by_id = {j.id: j for j in existing}

    if not LINKS_FILE.is_file():
        return 0, len(existing)

    urls = [line.strip() for line in LINKS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Allow blob URLs (e.g. .../blob/main/NEW_GRAD_USA.md); filter out internship-only repo names
    urls = [u for u in urls if "intern" not in u.lower()]

    # Collect new jobs per source so we can interleave and cap (ensures mix from each link)
    new_by_source: dict[str, List[JobEntry]] = {}

    for repo_url in urls:
        raw_url = _github_url_to_raw(repo_url)
        if not raw_url:
            continue
        md = _fetch_markdown(raw_url)
        if not md:
            continue
        # Repo name for display: use repo segment (e.g. 2026-Data-Analysis-New-Grad, NEW_GRAD_USA)
        parts = repo_url.rstrip("/").split("/")
        if "blob" in parts:
            # .../org/repo/blob/branch/file.md -> repo is at index -3
            repo_name = parts[parts.index("blob") - 1]
        else:
            repo_name = parts[-1].replace(".md", "") if parts else "unknown"
        jobs = _detect_format_and_parse(md, source_repo=repo_name)
        for job in jobs:
            if job.id in seen_ids:
                continue
            new_by_source.setdefault(repo_name, []).append(job)

    # Interleave across sources so a single run doesn't add only one source
    all_new_jobs: List[JobEntry] = []
    source_keys = list(new_by_source.keys())
    indices = {s: 0 for s in source_keys}
    while True:
        made_progress = False
        for src in source_keys:
            lst = new_by_source[src]
            i = indices[src]
            if i < len(lst):
                all_new_jobs.append(lst[i])
                indices[src] = i + 1
                made_progress = True
                if len(all_new_jobs) >= MAX_JOBS_PER_RUN:
                    break
        if not made_progress or len(all_new_jobs) >= MAX_JOBS_PER_RUN:
            break
    all_new_jobs = all_new_jobs[:MAX_JOBS_PER_RUN]

    if not all_new_jobs:
        return 0, len(existing)

    # We no longer generate per-job bullet suggestions here to avoid hallucinations.
    # The Golden Hand LaTeX pipeline (in the Streamlit app) is the authoritative source
    # for resume edits, based on your portfolio + resume + a specific JD.
    for job in all_new_jobs:
        seen_ids.add(job.id)
        existing_by_id[job.id] = job

    # Persist: sort by newest first (age_days asc, date_posted desc, added_at desc)
    new_list = sorted(existing_by_id.values(), key=_job_sort_key_newest_first)
    _save_seen_ids(seen_ids)
    _save_results(new_list)
    return len(all_new_jobs), len(new_list)


# Month name to number for date_posted sort (Jobright "Mar 11")
_MONTH_NAMES = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}


def _job_sort_key_newest_first(j: JobEntry) -> tuple:
    """Sort key so newest jobs first: age_days asc, then date_posted desc, then added_at desc."""
    # 1) age_days: lower = newer (0d before 1d before 5d)
    age = j.age_days if j.age_days is not None else 99999
    # 2) date_posted: parse "Mar 11" to (year, month, day), higher = newer
    date_sort = 0
    if getattr(j, "date_posted", None) and j.date_posted:
        parts = j.date_posted.strip().split()
        if len(parts) >= 2:
            try:
                month_str = parts[0].lower()[:3]
                day = int(parts[1].strip(".,"))
                month = _MONTH_NAMES.get(month_str, 0)
                year = datetime.utcnow().year
                date_sort = year * 10000 + month * 100 + day
            except (ValueError, IndexError):
                pass
    # 3) added_at: more recent = larger string (ISO)
    added = j.added_at or ""
    return (age, -date_sort, added)


def get_job_posting_url(job: JobEntry) -> str:
    """
    Return the URL to use for fetching JD and for "Open job posting".
    Old data may have company URL in job.url; raw_line contains the real job link (Job Title or Posting).
    """
    url = (job.url or "").strip()
    raw = (job.raw_line or "").strip()
    # Already a job posting URL (jobright or workday/greenhouse)
    if "jobright.ai" in url and "/jobs/" in url and "/info/" in url:
        return url
    if "myworkdayjobs.com" in url or "workday.com" in url or "greenhouse.io" in url or "icims.com" in url:
        return url
    if "wd" in url and "myworkday" in url:  # wd5.myworkdayjobs.com etc
        return url
    # Try to extract job posting URL from raw_line (jobright Job Title link or speedyapply Posting link)
    if "jobright.ai/jobs/info/" in raw:
        m = re.search(r"(https?://jobright\.ai/jobs/info/[^\s)\"]+)", raw)
        if m:
            return m.group(1).rstrip(".,;")
    if "href=" in raw:
        m = re.search(r'href="(https?://[^"]+)"[^>]*>\s*<img', raw, re.I)
        if m:
            return m.group(1)
    return url


def _is_valid_job_entry(j: JobEntry) -> bool:
    """Exclude placeholder/site links: jobright.ai homepage, or title is just 'jobright.ai'."""
    if (j.title or "").strip().lower() == "jobright.ai":
        return False
    if "jobright.ai" in (j.url or ""):
        if "/jobs/" not in j.url or "/info/" not in j.url:
            return False
    return True


def load_latest_results(limit: int = 10) -> List[JobEntry]:
    """
    Load the latest job scan results (up to `limit` jobs), newest first (by age/date posted).
    Excludes invalid entries (e.g. jobright.ai site link, not a job posting).
    """
    jobs = _load_existing_results()
    jobs = [j for j in jobs if _is_valid_job_entry(j)]
    jobs = sorted(jobs, key=_job_sort_key_newest_first)
    return jobs[:limit]


def load_latest_results_mixed(limit: int = 30) -> List[JobEntry]:
    """
    Load jobs from all sources, sorted newest first (Age / Date Posted ascending = newest first).
    Excludes invalid entries (e.g. jobright.ai site link). Returns up to `limit` jobs.
    """
    jobs = _load_existing_results()
    jobs = [j for j in jobs if _is_valid_job_entry(j)]
    if not jobs or limit <= 0:
        return jobs[:limit] if jobs else []
    jobs = sorted(jobs, key=_job_sort_key_newest_first)
    return jobs[:limit]

