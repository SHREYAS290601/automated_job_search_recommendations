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

MAX_JOBS_PER_RUN = 10


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
        return [JobEntry(**item) for item in data.get("jobs", [])]
    except Exception:
        return []


def _save_results(jobs: List[JobEntry]) -> None:
    _ensure_data_dir()
    RESULTS_FILE.write_text(
        json.dumps({"jobs": [asdict(j) for j in jobs]}, indent=2),
        encoding="utf-8",
    )


def _github_to_raw_readme(url: str) -> Optional[str]:
    """
    Convert https://github.com/org/repo to raw README URL.

    This is heuristic but works for the jobright-ai style repos.
    """
    m = re.match(r"https://github.com/([^/]+)/([^/#]+)", url.strip())
    if not m:
        return None
    org, repo = m.group(1), m.group(2)
    return f"https://raw.githubusercontent.com/{org}/{repo}/HEAD/README.md"


def _fetch_markdown(url: str, timeout: int = 10) -> Optional[str]:
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        return None
    return None


_INTERNSHIP_WORDS = ["intern", "internship", "co-op", "co op", "coop"]


def _is_entry_level_line(line: str) -> bool:
    lower = line.lower()
    if any(w in lower for w in _INTERNSHIP_WORDS):
        return False
    # Heuristics: look for entry level / new grad wording
    if "entry level" in lower or "new grad" in lower or "new-graduate" in lower:
        return True
    # Also allow repos that are clearly "New-Grad" even if line text doesn't say it
    return False


_LINK_RE = re.compile(r"\[(?P<title>.*?)\]\((?P<url>https?://[^\s)]+)\)")


def _parse_jobs(markdown: str, source_repo: str) -> List[JobEntry]:
    jobs: List[JobEntry] = []
    now = datetime.utcnow().isoformat()
    for line in markdown.splitlines():
        if "http" not in line:
            continue
        if not _is_entry_level_line(line):
            continue
        m = _LINK_RE.search(line)
        if not m:
            continue
        title = m.group("title").strip()
        url = m.group("url").strip()
        job_id = f"{source_repo}|{title}|{url}"
        jobs.append(
            JobEntry(
                id=job_id,
                title=title,
                url=url,
                source_repo=source_repo,
                raw_line=line.strip(),
                added_at=now,
            )
        )
    return jobs


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
    # Filter out obvious internship repos by name
    urls = [u for u in urls if "intern" not in u.lower()]

    all_new_jobs: List[JobEntry] = []

    for repo_url in urls:
        raw_url = _github_to_raw_readme(repo_url)
        if not raw_url:
            continue
        md = _fetch_markdown(raw_url)
        if not md:
            continue
        repo_name = repo_url.rstrip("/").split("/")[-1]
        jobs = _parse_jobs(md, source_repo=repo_name)
        for job in jobs:
            if job.id in seen_ids:
                continue
            all_new_jobs.append(job)

    if not all_new_jobs:
        return 0, len(existing)

    # Limit per run
    all_new_jobs = all_new_jobs[:MAX_JOBS_PER_RUN]

    # We no longer generate per-job bullet suggestions here to avoid hallucinations.
    # The Golden Hand LaTeX pipeline (in the Streamlit app) is the authoritative source
    # for resume edits, based on your portfolio + resume + a specific JD.
    for job in all_new_jobs:
        seen_ids.add(job.id)
        existing_by_id[job.id] = job

    # Persist
    new_list = sorted(existing_by_id.values(), key=lambda j: j.added_at, reverse=True)
    _save_seen_ids(seen_ids)
    _save_results(new_list)
    return len(all_new_jobs), len(new_list)


def load_latest_results(limit: int = 10) -> List[JobEntry]:
    """
    Load the latest job scan results (up to `limit` jobs), most recent first.
    """
    jobs = _load_existing_results()
    return jobs[:limit]

