import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from jobs_pipeline import get_job_posting_url, load_latest_results, load_latest_results_mixed, run_job_scan


load_dotenv()


class RewrittenBullet(BaseModel):
    original: str
    new: str


class ResumeReview(BaseModel):
    roast: str
    missing_keywords: List[str]
    rewritten_bullets: List[RewrittenBullet]
    overall_match_score: int
    skills_match_score: int
    impact_score: int
    coaching_notes: str


API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Please add it to your environment or .env file."
    )

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL or None,
)

PROJECT_ROOT = Path(__file__).resolve().parent

def _get_templates_dir() -> Path:
    """Return templates directory: try PROJECT_ROOT/templates, cwd/templates, env TEMPLATES_DIR or PROJECT_ROOT."""
    candidates = [
        PROJECT_ROOT / "templates",
        Path.cwd() / "templates",
    ]
    if os.getenv("TEMPLATES_DIR"):
        candidates.insert(0, Path(os.getenv("TEMPLATES_DIR")))
    if os.getenv("PROJECT_ROOT"):
        candidates.insert(0, Path(os.getenv("PROJECT_ROOT")) / "templates")
    for d in candidates:
        if d and d.is_dir():
            return d
    return PROJECT_ROOT / "templates"

def _resolve_template_path(filename: str) -> Path:
    """First path under templates dir that exists (for filename)."""
    templates_dir = _get_templates_dir()
    return templates_dir / filename

COVER_LETTER_SYSTEM = (
    "You are an expert career coach. Your task is to write a tailored cover letter for a candidate applying to a "
    "specific job. The user message will start with 'TARGET JOB (you MUST use this company and role—no other):' "
    "followed by one line (e.g. 'Data Scientist at Acme Corp'). You MUST use that exact company name and role for "
    "the salutation (e.g. 'Dear Acme Corp Hiring Team') and throughout the letter. Never use a different company or "
    "role. You are also given: (1) the candidate's resume text, (2) the full job description, and (3) a sample cover "
    "letter template. Generate a new cover letter that: preserves the structure, tone, and paragraph flow of the "
    "template; uses the company and role from the TARGET JOB line; and weaves in concrete achievements and skills "
    "from the resume that match the JD. Write in first person as the candidate. Output only the cover letter text—"
    "no headings, no meta-commentary."
)


def _load_cover_letter_template() -> tuple[str, str]:
    """Load cover letter template. Returns (content, path_used). Content empty if not found."""
    path = _resolve_template_path("cover_letter.txt")
    if not path.is_file():
        return "", str(path)
    return path.read_text(encoding="utf-8").strip(), str(path)


def _latex_to_plain_text(latex: str) -> str:
    """Rough strip of LaTeX to get plain text for LLM (e.g. roast)."""
    text = re.sub(r"\\[a-zA-Z]+\*?\{([^{}]*)\}", r"\1", latex)
    text = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^{}]*\})?", "", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\\%", "%", text)
    text = re.sub(r"\\&", "&", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_relevance_metric(resume_text: str, job_description: str, model_name: str) -> tuple[int, str]:
    """Return (score 0-100, one-sentence explanation) for resume vs JD relevance."""
    prompt = (
        "You are a recruiter. Given the resume text and job description below, output exactly two lines:\n"
        "Line 1: A single integer from 0 to 100 (relevance score).\n"
        "Line 2: One short sentence explaining the score.\n\n"
        "=== RESUME ===\n"
        f"{resume_text[:8000]}\n\n"
        "=== JOB DESCRIPTION ===\n"
        f"{job_description[:4000]}\n"
    )
    response = client.responses.create(
        model=model_name,
        input=[{"role": "user", "content": prompt}],
    )
    text = _extract_text_from_response(response).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    score = 50
    explanation = "Relevance assessed."
    if lines:
        first = re.sub(r"[^0-9]", "", lines[0])
        if first:
            score = min(100, max(0, int(first[:3])))
        if len(lines) > 1:
            explanation = lines[1]
    return score, explanation


def _job_header_from_jd(job_description: str) -> str:
    """First line or first 400 chars of JD—usually contains 'Role at Company' or 'Role, Company'. Use for targeting."""
    jd = (job_description or "").strip()
    if not jd:
        return ""
    first_line = jd.split("\n")[0].strip()
    return first_line if len(first_line) <= 400 else first_line[:397] + "..."


def get_cover_letter(resume_text: str, job_description: str, template: str, model_name: str) -> str:
    """Generate a tailored cover letter from resume, JD, and template."""
    if not template.strip():
        raise ValueError("Cover letter template is empty.")
    job_header = _job_header_from_jd(job_description)
    target_instruction = (
        f"TARGET JOB (you MUST use this company and role—no other): {job_header}\n\n"
        "The salutation must use the company name from the TARGET JOB line above (e.g. 'Dear [That Company] Hiring Team'). "
        "The role and company mentioned in the letter body must also match the TARGET JOB line exactly.\n\n"
    ) if job_header else ""
    user_content = (
        target_instruction
        + "Generate a tailored cover letter using the following inputs. The company name and role title must come "
        "ONLY from the TARGET JOB line above and the JOB DESCRIPTION below—never from any other source.\n\n"
        "=== RESUME ===\n"
        f"{resume_text}\n\n"
        "=== JOB DESCRIPTION ===\n"
        f"{job_description}\n\n"
        "=== SAMPLE COVER LETTER TEMPLATE (match its structure and tone) ===\n"
        f"{template}\n"
    )
    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": COVER_LETTER_SYSTEM},
            {"role": "user", "content": user_content},
        ],
    )
    return _extract_text_from_response(response).strip()


SYSTEM_PROMPT = """
You are an AI pipeline with two distinct but coordinated roles, evaluating any professional resume against any job description.

ROLE 1 – REALITY CHECK (ROAST)
- Act as a cynical, sleep-deprived senior recruiter.
- Brutally roast the provided resume against the provided job description.
- Call out specific corporate fluff, vague metrics, buzzwords without evidence, and any generic or irrelevant content.
- Explain clearly why this resume would be filtered out or rejected, referencing concrete examples from the resume.

ROLE 2 – GAME PLAN (WHAT TO DO)
- Switch into an elite career coach.
- Identify the most important missing or under-emphasized technical and role-relevant keywords from the job description.
- Select the 3 weakest bullet points from the resume and rewrite them using the XYZ formula:
  “Accomplished [X] as measured by [Y], by doing [Z]”
  so that they are sharply aligned to the job description.
- Provide a practical, step-by-step improvement plan the candidate can follow to upgrade this resume for this job.

OUTPUT CONTRACT (MUST MATCH SCHEMA)
- roast (string): A direct, detailed critique in paragraph form. It should be honest, specific, and pointed, but still professional.
- missing_keywords (list[string]): 5–10 high-impact, role-relevant keywords or phrases from the job description that should appear more strongly in the resume.
- rewritten_bullets (list[object]): Exactly 3 objects, each with:
  - original (string): the original weak bullet (quote or closely paraphrase from the resume).
  - new (string): a rewritten bullet using the XYZ formula.
- overall_match_score (int): 0–100. Overall fit of the resume to the job description.
- skills_match_score (int): 0–100. How well the skills, tools, and domain knowledge match.
- impact_score (int): 0–100. How strong, specific, and outcome-focused the bullets are.
- coaching_notes (string): 5–8 numbered, concrete steps summarizing what the candidate should do next to improve this resume for this job (structure, content, keywords, and focus).

IMPORTANT
- Your output will be parsed into a strict JSON schema using these fields. Do not include any extra top-level fields.
- Always respect the requested roast tone passed in the user message (Brutal, Balanced, Gentle) while staying honest and useful.
"""


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract raw text from an uploaded PDF using PyMuPDF."""
    if uploaded_file is None:
        return ""

    file_bytes = uploaded_file.read()
    if not file_bytes:
        return ""

    text_chunks: List[str] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text_chunks.append(page_text)

    return "\n".join(text_chunks).strip()


def _extract_text_from_response(response) -> str:
    """Extract assistant message text from Responses API response (no response_format)."""
    if not getattr(response, "output", None):
        return ""
    parts = []
    for item in response.output:
        if getattr(item, "type", None) == "message" and getattr(item, "content", None):
            for block in item.content:
                if getattr(block, "type", None) == "output_text" and getattr(block, "text", None):
                    parts.append(block.text)
    return "\n".join(parts) if parts else ""


def _is_likely_marketing_page(text: str) -> bool:
    """True if content looks like a company marketing/landing page, not a job description."""
    if not text or len(text) < 500:
        return False
    lower = text.lower()
    # Marketing page indicators (company homepage, product pitch)
    if "get started" in lower and "learn more" in lower:
        if lower.count("learn more") >= 2 or lower.count("get started") >= 2:
            return True
    if "unlock profit" in lower or "supply chain runs on" in lower:
        return True
    if "schedule a demo" in lower and "contact" in lower and "responsibilities" not in lower:
        return True
    # Job page indicators (if present, likely a JD)
    if "responsibilities" in lower and ("qualification" in lower or "requirements" in lower):
        return False
    if "job description" in lower or "apply now" in lower:
        return False
    return False


def _extract_jobright_jd(soup: BeautifulSoup) -> str | None:
    """Extract JD from jobright.ai job page: from job title through Benefits, before Company section."""
    for tag in soup.find_all(["script", "style", "nav", "footer", "noscript", "iframe"]):
        tag.decompose()
    # Jobright: look for section headings (Responsibilities, Qualification, Benefits) and take that block
    parts = []
    stop_headers = re.compile(r"^\s*(Company|H1B|Funding|Leadership|Recent News)\s*$", re.I)
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = el.get_text(separator=" ", strip=True)
        if not text or len(text) < 3:
            continue
        if el.name in ("h2", "h3", "h4") and stop_headers.match(text):
            break
        parts.append(text)
    if parts:
        jd = "\n\n".join(parts)
        if len(jd) > 200:
            return jd
    return None


def _extract_jd_from_soup(soup: BeautifulSoup, request_url: str = "") -> str | None:
    """Traverse the page with bs4 and extract job description from the job posting link only."""
    # Remove script, style, nav, footer to reduce noise
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()

    # Jobright.ai: use structure-based extraction (job title → Responsibilities → Qualification → Benefits)
    if "jobright.ai" in request_url and "/jobs/" in request_url:
        jd = _extract_jobright_jd(soup)
        if jd:
            return jd

    # Workday and other ATS: known selectors first
    selectors = [
        '[data-automation-id="jobPostingDescription"]',  # Workday
        '[data-qa="job-description"]',
        '.job-description',
        '.job-description__content',
        '.job-posting-description',
        '[class*="JobDescription"]',
        '[class*="job-description"]',
        '[class*="jobDescription"]',
        '[id*="job-description"]',
        '[id*="jobDescription"]',
        '.job-details',
        '[data-cy="job-description"]',
    ]
    for sel in selectors:
        try:
            el = soup.select_one(sel)
            if el:
                text = el.get_text(separator="\n", strip=True)
                if text and len(text) > 200 and not _is_likely_marketing_page(text):
                    return text
        except Exception:
            continue
    # Generic fallbacks (avoid main/article which often grab whole page including marketing)
    for sel in ["main", "article", "[role='main']", ".content__body", ".description"]:
        try:
            el = soup.select_one(sel)
            if el:
                text = el.get_text(separator="\n", strip=True)
                if text and len(text) > 300 and not _is_likely_marketing_page(text):
                    return text
        except Exception:
            continue
    body = soup.find("body")
    if body:
        text = body.get_text(separator="\n", strip=True)
        if text and len(text) > 100 and not _is_likely_marketing_page(text):
            return text
    return None


def _fetch_jd(url: str) -> tuple[str | None, str | None]:
    """
    Fetch the job description from the exact job posting URL.
    - Jobright (1st source): JD is at the Job Title column link (jobright.ai/jobs/info/...).
    - Speedyapply (2nd source): JD is at the Posting column link (Apply button href).
    Rejects content if we were redirected to a company homepage or page looks like marketing.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = requests.get(url, timeout=20, headers=headers, allow_redirects=True)
        if resp.status_code != 200:
            return None, None
        # If we were redirected to a different host (e.g. company site), only use if content looks like a JD
        from urllib.parse import urlparse
        requested_host = urlparse(url).netloc.lower()
        final_host = urlparse(resp.url).netloc.lower()
        soup = BeautifulSoup(resp.text, "html.parser")
        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        text = _extract_jd_from_soup(soup, request_url=url)
        if not text:
            text = soup.get_text(separator="\n", strip=True)
        if text and len(text) < 100:
            return None, title
        # If redirected to different domain and content looks like marketing, reject
        if requested_host != final_host and _is_likely_marketing_page(text):
            return None, title
        if text and _is_likely_marketing_page(text):
            return None, title
        return (text or None), title
    except Exception:
        return None, None


def _ensure_noindent_before_roles(latex: str) -> str:
    """Ensure \\noindent appears before each company/role line (\\textbf{...\\hfill...\\\\)."""
    lines = latex.split("\n")
    out: List[str] = []
    for i, line in enumerate(lines):
        # Company/role line: \textbf{...} \hfill ... \\
        if re.match(r"^\s*\\textbf\s*\{[^}]*\}\s*\\hfill", line):
            prev = (out[-1].strip() if out else "")
            if prev != "\\noindent":
                out.append("\\noindent")
        out.append(line)
    return "\n".join(out)


def _ensure_data_science_intern_three_bullets(
    latex: str, portfolio_text: str, job_description: str, model_name: str
) -> str:
    """If the Data Science Intern role has only 2 \\item, add a third (LLM or fallback)."""
    # Find the LAST "Data Science Intern" (earliest/most junior role is usually last in the list)
    idx = latex.rfind("Data Science Intern")
    if idx == -1:
        return latex
    # The itemize for this role is the next \begin{itemize} after the role title
    begin_idx = latex.find("\\begin{itemize}", idx)
    if begin_idx == -1:
        return latex
    end_idx = latex.find("\\end{itemize}", begin_idx)
    if end_idx == -1:
        return latex
    block = latex[begin_idx:end_idx]
    item_count = len(re.findall(r"\\item\s*", block))
    if item_count >= 3:
        return latex
    # Build third bullet: try LLM, else use fallback
    third_text = ""
    bullets = re.findall(r"\\item\s+(.+?)(?=\\item|$)", block, re.DOTALL)
    two_text = "\n".join((b[:300].strip() for b in bullets[:2]))
    # Provide a small slice of portfolio for grounding (avoid hallucinated metrics)
    pt = (portfolio_text or "").strip()
    pt_snip = pt[:2500]
    if "Data Science Intern" in pt:
        i = pt.rfind("Data Science Intern")
        pt_snip = pt[max(0, i - 800) : i + 1700]
    prompt = (
        "You are completing a Data Science Intern resume entry.\n"
        "Write EXACTLY ONE additional bullet in the same tone and XYZ style.\n"
        "HARD CONSTRAINTS:\n"
        "- Do NOT invent any numbers, percentages, counts, latency, costs, model names, datasets, or metrics.\n"
        "- You may ONLY reuse facts/metrics that already appear in the bullets below or in the PORTFOLIO SNIPPET.\n"
        "- If you cannot find a metric to support the claim, write a qualitative bullet WITHOUT any numbers.\n"
        "- One line only. No \\item, no LaTeX.\n\n"
        "EXISTING BULLETS:\n"
        + two_text
        + "\n\nPORTFOLIO SNIPPET:\n"
        + pt_snip
        + "\n\nJOB DESCRIPTION (keywords only):\n"
        + job_description[:800]
    )
    try:
        response = client.responses.create(
            model=model_name,
            input=[{"role": "user", "content": prompt}],
        )
        raw = _extract_text_from_response(response).strip()
        third_text = raw.split("\n")[0].strip()
        third_text = re.sub(r"^[\s\\item\-\.\d]*", "", third_text)
    except Exception:
        pass
    if not third_text or len(third_text) < 15:
        third_text = (
            "Strengthened data pipelines and analytics quality, by applying Python, SQL, and "
            "document processing to support stakeholder reporting and decision-making."
        )
    # Escape % for LaTeX
    third_text = third_text.replace("%", "\\%")
    new_item = " \\item " + third_text + "\n"
    return latex[:end_idx] + new_item + latex[end_idx:]


def _clean_latex_experience(latex: str) -> str:
    """Remove verbatim/code wrappers, (X)/(Y)/(Z) labels, and trailing meta-comments from LaTeX experience section."""
    text = latex.strip()
    # Strip \begin{verbatim} ... \end{verbatim} (and lstlisting)
    for env in ("verbatim", "lstlisting"):
        beg = f"\\begin{{{env}}}"
        end = f"\\end{{{env}}}"
        if beg in text and end in text:
            i, j = text.find(beg), text.find(end) + len(end)
            text = text[:i] + text[j:]
            text = text.strip()
    # Remove trailing meta-comments like " % added JD-focused bullet" (must do before escaping %)
    text = re.sub(r"\s*%\s*added[^\n\\]*", "", text, flags=re.IGNORECASE)
    # Remove lines that are only LaTeX comments
    text = re.sub(r"^\s*%.*$", "", text, flags=re.MULTILINE)
    # Remove literal (X), (Y), (Z) labels so bullets read as natural prose
    text = re.sub(r"\s*\([XYZ]\)\s*([,.]?)\s*", r" \1", text)
    text = re.sub(r"\s*\[[XYZ]\]\s*([,.]?)\s*", r" \1", text)
    text = re.sub(r"  +", " ", text)
    # In LaTeX, % starts a comment and truncates the line—escape so "50%" renders fully
    text = re.sub(r"(?<!\\)%", r"\\%", text)
    return text.strip()


def _parse_golden_hand_rationale(rationale_block: str) -> dict:
    """Parse the text before LATEX_EXPERIENCE into compatibility_rating, rationale, alterations."""
    out: dict = {"compatibility_rating": "", "rationale": "", "alterations": "", "raw": rationale_block or ""}
    if not (rationale_block or rationale_block.strip()):
        return out
    raw = rationale_block.strip()
    # COMPATIBILITY_RATING: may be on same line or next line (e.g. "4.8 / 5.0 (High Likelihood of Callback)")
    if "COMPATIBILITY_RATING:" in raw:
        rest = raw.split("COMPATIBILITY_RATING:", 1)[1]
        for line in rest.split("\n"):
            line = line.strip()
            # Skip empty, section headers, or junk like "**" (model markdown artifact)
            if not line or line.startswith("RATIONALE:") or line.startswith("ALTERATIONS:"):
                continue
            if line.replace("*", "").strip() == "" or len(line.replace("*", "").strip()) < 4:
                continue
            # Should look like a rating: e.g. "4.8 / 5.0 (...)" — has a digit and usually /
            if re.search(r"\d", line) and ("/" in line or "5" in line or "out of" in line.lower()):
                out["compatibility_rating"] = line.replace("**", "").strip() or line
                break
            # Else accept if it's a substantial line (not just **)
            if len(line) >= 6:
                out["compatibility_rating"] = line.replace("**", "").strip() or line
                break
    # RATIONALE: ... (until ALTERATIONS: or XYZ_FORMAT_EXAMPLES:)
    if "RATIONALE:" in raw:
        rest = raw.split("RATIONALE:", 1)[1]
        for sep in ["ALTERATIONS:", "XYZ_FORMAT_EXAMPLES:", "LATEX_EXPERIENCE:"]:
            if sep in rest:
                rest = rest.split(sep)[0]
        out["rationale"] = rest.strip()
    # ALTERATIONS: ... (until XYZ_FORMAT_EXAMPLES: or end)
    if "ALTERATIONS:" in raw:
        rest = raw.split("ALTERATIONS:", 1)[1]
        if "XYZ_FORMAT_EXAMPLES:" in rest:
            rest = rest.split("XYZ_FORMAT_EXAMPLES:")[0]
        out["alterations"] = rest.strip()
    return out


def _clean_rationale_display(text: str) -> str:
    """Remove stray markdown ** so rationale/alterations don't show literal asterisks."""
    if not text:
        return text
    t = text.strip()
    # Strip leading ** (model sometimes outputs **Your... with no closing)
    while t.startswith("**") and not t.startswith("****"):
        t = t[2:].lstrip()
    while t.endswith("**") and not t.endswith("****"):
        t = t[:-2].rstrip()
    return t


def _sanitize_bullet_claims(bullet: str, allowed_text: str) -> str:
    """
    Remove unsupported (hallucinated) factual claims from a single bullet.
    Strategy:
    - If a number appears in the bullet but that exact number-token does not appear anywhere in allowed_text,
      strip numeric tokens and measurement units and remove dangling "as measured by ..." phrases.
    - If a distinctive model/tool token (e.g. Falcon-7B, Qwen3-32B) appears but not in allowed_text, remove it.
    This intentionally prefers slightly-generic but truthful bullets over impressive-but-false metrics.
    """
    if not bullet:
        return bullet
    src = (allowed_text or "")
    b = bullet.strip()

    # Remove unsupported named model/tool tokens (keep conservative: tokens with digits + letters and hyphen)
    # Example: Qwen3-32B, Falcon-7B, LLaMA, LoRA/QLoRA, Pix2Pix, ConvNext
    # We only remove if the exact token (case-insensitive) doesn't appear in the sources.
    token_candidates = set(re.findall(r"\b[A-Za-z][A-Za-z0-9/+.-]{2,}\b", b))
    src_lower = src.lower()
    for tok in sorted(token_candidates, key=len, reverse=True):
        # Skip common English / resume words
        if tok.lower() in {
            "and",
            "the",
            "for",
            "with",
            "by",
            "to",
            "of",
            "in",
            "on",
            "as",
            "a",
            "an",
            "sql",
            "python",
            "azure",
            "powerbi",
            "tensorflow",
            "pytorch",
            "llm",
            "rag",
            "nlp",
        }:
            continue
        # Only consider "distinctive" tokens that are likely to be specific claims
        if not (re.search(r"\d", tok) or "/" in tok or "-" in tok or tok.isupper()):
            continue
        if tok.lower() not in src_lower:
            b = re.sub(rf"\b{re.escape(tok)}\b", "", b)

    # If bullet contains numbers, only keep those numbers that exist in source text.
    nums = re.findall(r"\d[\d,\.]*", b)
    unsupported = []
    for n in nums:
        if n not in src:
            unsupported.append(n)
    if unsupported:
        # Strip all numeric tokens and common adjacent units/markers
        b = re.sub(
            r"(\b\d[\d,\.]*\+?\b)\s*(?:\\%|%|ms|s|sec|secs|seconds|mins|minutes|hrs|hours|k|K|m|M|b|B)?",
            "",
            b,
        )
        # Remove arrow ranges like "1 s → 200 ms" (after numbers stripped, clean leftovers)
        b = b.replace("→", " ")
        b = re.sub(r"\(\s*\)", "", b)
        # Remove dangling metric phrases that often follow invented numbers
        b = re.sub(r"\bas measured by\b[^,.;]*", "", b, flags=re.I)
        b = re.sub(r"\bwhile maintaining\b[^,.;]*", "", b, flags=re.I)
        b = re.sub(r"\bguaranteeing\b[^,.;]*", "", b, flags=re.I)
        b = re.sub(r"\bimproving\b[^,.;]*\bscore[s]?\b", "improving model explainability", b, flags=re.I)

    # Final cleanup
    b = re.sub(r"\s{2,}", " ", b).strip()
    b = re.sub(r"\s+([,.;:])", r"\1", b)
    b = re.sub(r"^[\-\u2022•\s]+", "", b).strip()
    if b and not b.endswith("."):
        b += "."
    return b


def _sanitize_experience_section(latex_experience: str, allowed_text: str) -> str:
    """Sanitize every \\item bullet inside a LaTeX experience section."""
    if not latex_experience:
        return latex_experience

    def _repl(m: re.Match) -> str:
        content = m.group(1).strip()
        cleaned = _sanitize_bullet_claims(content, allowed_text=allowed_text)
        return "\\item " + cleaned

    return re.sub(r"\\item\s+(.+?)(?=(?:\n\s*\\item|\n\s*\\end\{itemize\}|$))", _repl, latex_experience, flags=re.DOTALL)


def _merge_experience_into_tex(full_tex: str, new_experience: str) -> str:
    """Replace the Professional Experience section in full_tex with new_experience."""
    # Match common section headers (with or without *)
    section_pattern = re.compile(
        r"(\\(?:section|section\*)\s*\{[^}]*Professional\s+Experience[^}]*\})",
        re.IGNORECASE,
    )
    match = section_pattern.search(full_tex)
    if not match:
        # Fallback: any section with "Experience"
        section_pattern = re.compile(
            r"(\\(?:section|section\*)\s*\{[^}]*Experience[^}]*\})",
            re.IGNORECASE,
        )
        match = section_pattern.search(full_tex)
    if not match:
        return full_tex
    start = match.start()
    # Find next \section or \end{document}
    rest = full_tex[start:]
    next_section = re.search(r"\n\s*\\(?:section|section\*)\s*\{", rest[1:])
    end_doc = rest.find("\\end{document}")
    if end_doc == -1:
        end_doc = len(rest)
    if next_section:
        end = next_section.start() + 1
    else:
        end = end_doc
    if end <= 0:
        end = len(rest)
    before = full_tex[:start]
    after = rest[end:]
    return before + new_experience.strip() + "\n\n" + after


def _extract_experience_section(full_tex: str) -> str:
    """Extract the Experience section from a full LaTeX resume (best-effort)."""
    # Match common section headers (with or without *)
    section_pattern = re.compile(
        r"(\\(?:section|section\*)\s*\{[^}]*Professional\s+Experience[^}]*\})",
        re.IGNORECASE,
    )
    match = section_pattern.search(full_tex)
    if not match:
        section_pattern = re.compile(
            r"(\\(?:section|section\*)\s*\{[^}]*Experience[^}]*\})",
            re.IGNORECASE,
        )
        match = section_pattern.search(full_tex)
    if not match:
        return ""
    start = match.start()
    rest = full_tex[start:]
    next_section = re.search(r"\n\s*\\(?:section|section\*)\s*\{", rest[1:])
    end_doc = rest.find("\\end{document}")
    if end_doc == -1:
        end_doc = len(rest)
    if next_section:
        end = next_section.start() + 1
    else:
        end = end_doc
    if end <= 0:
        end = len(rest)
    return rest[:end].strip()


def _parse_experience_role_blocks(experience_tex: str) -> list[dict]:
    """
    Parse an Experience section into role blocks.
    Each block contains:
      - header: the company/date line + title line (as in LaTeX)
      - itemize_opts: itemize options string (inside [...]) if present
      - bullets: list of bullet strings (LaTeX content after \\item)
      - block_start, block_end: indices in experience_tex for the whole role block (header + itemize)
      - itemize_start, itemize_end: indices for the itemize content range
    """
    if not experience_tex:
        return []
    blocks: list[dict] = []

    # Identify each role by the company line: \textbf{...} \hfill ... \\
    # Then title line usually: \textbf{\underline{...}}
    company_re = re.compile(r"^\s*\\textbf\s*\{[^}]+\}\s*\\hfill\s*.+?\\\\\s*$", re.MULTILINE)
    matches = list(company_re.finditer(experience_tex))
    for mi, m in enumerate(matches):
        block_start = m.start()
        block_end = matches[mi + 1].start() if mi + 1 < len(matches) else len(experience_tex)
        block_text = experience_tex[block_start:block_end]

        # Find itemize inside this block
        begin_m = re.search(r"\\begin\{itemize\}(\[[^\]]*\])?", block_text)
        end_m = re.search(r"\\end\{itemize\}", block_text)
        if not begin_m or not end_m:
            continue
        itemize_opts = begin_m.group(1) or ""
        itemize_start = block_start + begin_m.end()
        itemize_end = block_start + end_m.start()

        # Header is everything from block start through the begin{itemize} line
        header = block_text[: begin_m.end()].rstrip()
        # Extract bullets
        itemize_body = experience_tex[itemize_start:itemize_end]
        bullets = [b.strip() for b in re.findall(r"\\item\s+(.+?)(?=(?:\n\s*\\item|$))", itemize_body, flags=re.DOTALL)]
        blocks.append(
            {
                "header": header,
                "itemize_opts": itemize_opts,
                "bullets": bullets,
                "block_start": block_start,
                "block_end": block_end,
                "itemize_start": itemize_start,
                "itemize_end": itemize_end,
            }
        )
    return blocks


def _rewrite_role_bullets_strict(
    role_header: str,
    bullets: list[str],
    portfolio_text: str,
    job_description: str,
    model_name: str,
) -> list[str]:
    """
    Rewrite bullets for ONE role only. Returns same number of bullets.
    Hard constraints: do not introduce facts/metrics/tools not present in role bullets or portfolio text.
    """
    if not bullets:
        return bullets

    # Keep prompt small and grounded: provide role header + bullets + relevant portfolio slice
    pt = (portfolio_text or "").strip()
    # Heuristic: if role title string appears, slice around it; else just take top chunk
    role_hint = ""
    title_m = re.search(r"\\underline\{([^}]+)\}", role_header)
    if title_m:
        role_hint = title_m.group(1).strip()
    pt_snip = pt[:3000]
    if role_hint and role_hint in pt:
        i = pt.find(role_hint)
        pt_snip = pt[max(0, i - 1200) : i + 2000]

    # Bullet source text for "allowed tokens" check
    allowed_local = ("\n".join(bullets) + "\n\n" + pt_snip).lower()

    prompt = (
        "You are rewriting resume bullets for EXACTLY ONE role. You MUST NOT mix content from other roles.\n\n"
        "ROLE HEADER (do not change):\n"
        f"{role_header}\n\n"
        "BULLETS (authoritative; keep same count, same role; rewrite wording only):\n"
        + "\n".join([f"- {b}" for b in bullets])
        + "\n\n"
        "PORTFOLIO SNIPPET (only use facts found here):\n"
        + pt_snip
        + "\n\n"
        "JOB DESCRIPTION (keywords to emphasize):\n"
        + job_description[:1200]
        + "\n\n"
        "HARD CONSTRAINTS:\n"
        "- Output MUST be valid JSON only: {\"bullets\": [\"...\", ...]}.\n"
        "- Return exactly "
        + str(len(bullets))
        + " bullets.\n"
        "- Do NOT add/remove/reorder bullets.\n"
        "- Do NOT add ANY new numbers, percentages, counts, timings, or benchmarks.\n"
        "- Do NOT add any new tools/technologies/models unless they appear in the BULLETS or PORTFOLIO SNIPPET.\n"
        "- Do NOT reference other companies or other roles.\n"
        "- Keep LaTeX-safe text (escape % as \\%).\n"
        "- Prefer expanding acronyms once (e.g. 'Retrieval Augmented Generation (RAG)') if relevant and supported.\n"
    )

    try:
        response = client.responses.create(
            model=model_name,
            input=[{"role": "user", "content": prompt}],
        )
        text = _extract_text_from_response(response).strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)
        data = json.loads(text)
        out_bullets = data.get("bullets") if isinstance(data, dict) else None
        if not isinstance(out_bullets, list) or len(out_bullets) != len(bullets):
            return bullets
        cleaned: list[str] = []
        for ob in out_bullets:
            if not isinstance(ob, str):
                cleaned.append("")
                continue
            s = ob.strip()
            s = s.replace("%", "\\%")
            s = re.sub(r"\s{2,}", " ", s)
            # If model tried to inject unseen token with digits/hyphens, strip it (local guard)
            toks = re.findall(r"\b[A-Za-z][A-Za-z0-9/+.-]{2,}\b", s)
            for tok in toks:
                if (re.search(r"\d", tok) or "-" in tok or "/" in tok) and tok.lower() not in allowed_local:
                    s = re.sub(rf"\b{re.escape(tok)}\b", "", s)
            s = re.sub(r"\s{2,}", " ", s).strip()
            cleaned.append(s)
        # Fallback to originals if model returned empty bullets
        if any(len(c.strip()) < 10 for c in cleaned):
            return bullets
        return cleaned
    except Exception:
        return bullets


def _rewrite_experience_section_strict(
    experience_tex: str,
    portfolio_text: str,
    job_description: str,
    model_name: str,
) -> tuple[str, list[tuple[str, str]]]:
    """
    Rewrite experience section by rewriting bullets per-role (prevents cross-contamination).
    Returns (new_experience_tex, changed_pairs[(original, optimized)]) for rationale generation.
    """
    blocks = _parse_experience_role_blocks(experience_tex)
    if not blocks:
        return experience_tex, []

    # Build new text by slicing original string and replacing itemize bodies
    out_parts: list[str] = []
    cursor = 0
    changed: list[tuple[str, str]] = []

    for b in blocks:
        out_parts.append(experience_tex[cursor : b["itemize_start"]])
        new_bullets = _rewrite_role_bullets_strict(
            role_header=b["header"],
            bullets=b["bullets"],
            portfolio_text=portfolio_text,
            job_description=job_description,
            model_name=model_name,
        )
        # Track a few changed bullet pairs for ALTERATIONS
        for old, new in zip(b["bullets"], new_bullets):
            if old.strip() != new.strip():
                changed.append((old.strip(), new.strip()))

        # Rebuild itemize body with same indentation as original
        rebuilt = "\n"
        for nb in new_bullets:
            rebuilt += " \\item " + nb.strip() + "\n"
        out_parts.append(rebuilt)
        out_parts.append(experience_tex[b["itemize_end"] : b["block_end"]])
        cursor = b["block_end"]

    out_parts.append(experience_tex[cursor:])
    return "".join(out_parts), changed


def _compile_tex_to_pdf(tex_content: str) -> bytes | None:
    """Compile LaTeX string to PDF; returns PDF bytes or None if pdflatex unavailable/fails."""
    if not shutil.which("pdflatex"):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        main = path / "main.tex"
        main.write_text(tex_content, encoding="utf-8")
        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                cwd=tmp,
                capture_output=True,
                timeout=60,
            )
            pdf_path = path / "main.pdf"
            if pdf_path.exists():
                return pdf_path.read_bytes()
        except (subprocess.TimeoutExpired, OSError):
            pass
    return None


def _golden_hand_from_text(
    portfolio_text: str,
    latex_source: str,
    job_description: str,
    model_name: str,
) -> tuple[str, str]:
    """
    Run the Golden Hand pipeline for an arbitrary job description.

    Returns (latex_experience, rationale_text).
    """
    # Strict approach: rewrite bullets per role-block (prevents cross-contamination),
    # then generate rationale/compatibility from the diffs.
    current_experience = _extract_experience_section(latex_source) or ""
    if not current_experience:
        raise ValueError("Could not find an Experience section in the LaTeX resume.")

    rewritten_experience, changed_pairs = _rewrite_experience_section_strict(
        current_experience,
        portfolio_text=portfolio_text,
        job_description=job_description,
        model_name=model_name,
    )
    latex_experience = rewritten_experience.strip()
    rationale_text = ""

    # Generate compatibility/rationale/alterations (no LaTeX) from diffs
    pairs_snip = changed_pairs[:5]
    diff_lines = "\n".join(
        [f"- Original: \"{o}\"\n  Optimized: \"{n}\"" for (o, n) in pairs_snip]
    )
    rationale_prompt = (
        "You are an ATS-focused resume coach. Given the job description and a few bullet rewrites, produce:\n\n"
        "COMPATIBILITY_RATING:\n"
        "One line: X.X / 5.0 (Label)\n\n"
        "RATIONALE:\n"
        "1–2 paragraphs on why the profile fits this JD.\n\n"
        "ALTERATIONS:\n"
        "For each pair below, add a 'Why this improves it' line (1–2 sentences).\n\n"
        "XYZ_FORMAT_EXAMPLES:\n"
        "2 short plain-text examples.\n\n"
        "IMPORTANT: Do not invent new facts/metrics; only explain the rewrite.\n\n"
        "=== JOB DESCRIPTION ===\n"
        + job_description[:2500]
        + "\n\n=== BULLET REWRITES ===\n"
        + (diff_lines or "- (No bullet changes detected)")
        + "\n"
    )
    try:
        rr = client.responses.create(model=model_name, input=[{"role": "user", "content": rationale_prompt}])
        rationale_text = _extract_text_from_response(rr).strip()
    except Exception:
        rationale_text = ""

    latex_experience = _clean_latex_experience(latex_experience)
    latex_experience = _ensure_noindent_before_roles(latex_experience)
    latex_experience = _ensure_data_science_intern_three_bullets(
        latex_experience,
        portfolio_text=portfolio_text,
        job_description=job_description,
        model_name=model_name,
    )
    # Anti-hallucination pass: strip any metrics/claims not present in the user's sources.
    # IMPORTANT: ignore commented-out LaTeX lines so old experiments in % comments do not reappear.
    cleaned_latex = re.sub(r"^\s*%.*$", "", latex_source or "", flags=re.MULTILINE)
    allowed_text = (portfolio_text or "") + "\n\n" + cleaned_latex
    latex_experience = _sanitize_experience_section(latex_experience, allowed_text=allowed_text)
    return latex_experience, rationale_text


def get_resume_review(
    resume_text: str,
    job_description: str,
    roast_tone: str,
    model_name: str,
) -> ResumeReview:
    """Call OpenAI with a structured Pydantic schema to review the resume."""
    if not resume_text.strip():
        raise ValueError("Resume text is empty.")
    if not job_description.strip():
        raise ValueError("Job description text is empty.")

    user_prompt = (
        "You are given a candidate resume and a target job description.\n\n"
        "=== RESUME ===\n"
        f"{resume_text}\n\n"
        "=== JOB DESCRIPTION ===\n"
        f"{job_description}\n\n"
        "Tone preference for the roast: "
        f"{roast_tone}.\n\n"
        "You must respond with ONLY a single JSON object (no markdown, no code fence) that matches this schema: "
        "roast (string), missing_keywords (list of strings), rewritten_bullets (list of {original, new}), "
        "overall_match_score (int 0-100), skills_match_score (int 0-100), impact_score (int 0-100), coaching_notes (string)."
    )

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.strip(),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    text = _extract_text_from_response(response)
    if not text or not text.strip():
        raise ValueError("Model returned no text.")

    # Strip optional markdown code block
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)

    data = json.loads(text)
    return ResumeReview.model_validate(data)


def main() -> None:
    st.set_page_config(
        page_title="Reality Check ATS",
        page_icon="🛑",
        layout="wide",
    )

    # Resolve template paths once before any page render
    templates_dir = _get_templates_dir()
    if "template_checks" not in st.session_state:
        # Handle case differences between macOS (case-insensitive) and Linux (case-sensitive)
        portfolio_path = templates_dir / "portfolio.pdf"
        if not portfolio_path.is_file():
            alt_portfolio = templates_dir / "Portfolio.pdf"
            if alt_portfolio.is_file():
                portfolio_path = alt_portfolio

        resume_tex_path = templates_dir / "resume.tex"
        if not resume_tex_path.is_file():
            alt_resume = templates_dir / "Resume.tex"
            if alt_resume.is_file():
                resume_tex_path = alt_resume

        st.session_state["template_checks"] = {
            "cover_letter": templates_dir / "cover_letter.txt",
            "portfolio": portfolio_path,
            "resume_tex": resume_tex_path,
            "resume_pdf": templates_dir / "resume.pdf",
        }

    st.title("Reality Check ATS 🛑")
    st.subheader(
        "Tailor your resume with The Golden Hand, scan live New Grad roles, and auto-generate resume tweaks."
    )

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Page",
            ["The Golden Hand", "Job Scanner", "Cover Letter"],
            index=0,
        )

        st.header("Global Input")
        job_description = st.text_area(
            "Paste the Job Description (optional for Golden Hand, required for roasting / relevance / cover letter)",
            height=260,
            placeholder="Paste the full job description here...",
            key="job_description_input",
        )

        st.divider()
        model_name = st.selectbox(
            "Model",
            ["gpt-oss-120b", "gpt-4.1-mini", "gpt-4.1", "gpt-4.1-preview"],
            index=0,
            help="Choose the model to use.",
        )

    # The Golden Hand page – LaTeX-focused resume builder.
    if page == "The Golden Hand":
        tc = st.session_state["template_checks"]
        has_portfolio = tc["portfolio"].is_file()
        has_resume_tex = tc["resume_tex"].is_file()

        missing = []
        if not has_portfolio:
            missing.append("portfolio.pdf")
        if not has_resume_tex:
            missing.append("resume.tex")
        if missing:
            st.warning(
                f"Place **{'** and **'.join(missing)}** in `templates/` so the app can load them automatically. "
                f"Looking in: `{tc['portfolio'].parent}`."
            )

        st.markdown("### The Golden Hand – JD-Specific LaTeX Builder")
        st.write(
            "Portfolio and LaTeX resume are loaded from **templates/** when present. Paste a job description in the "
            "sidebar and click Generate to get the updated Experience section."
        )

        portfolio_text: str = ""
        latex_source: str = ""
        if has_portfolio and has_resume_tex:
            portfolio_text = extract_text_from_pdf(BytesIO(tc["portfolio"].read_bytes()))
            if not portfolio_text:
                st.error("Could not extract text from `templates/portfolio.pdf`.")
            latex_source = tc["resume_tex"].read_text(encoding="utf-8", errors="ignore")
            st.success(f"Using **templates/portfolio.pdf** and **templates/resume.tex**.")
        else:
            portfolio_pdf = st.file_uploader(
                "Upload your Portfolio PDF (or add templates/portfolio.pdf)",
                type=["pdf"],
                key="portfolio_pdf",
            )
            latex_file = st.file_uploader(
                "Upload your LaTeX Resume (or add templates/resume.tex)",
                type=["tex"],
                key="latex_tex",
            )
            if portfolio_pdf:
                portfolio_text = extract_text_from_pdf(portfolio_pdf)
            if latex_file:
                try:
                    latex_source = latex_file.read().decode("utf-8", errors="ignore")
                except Exception:
                    st.error("Failed to read the uploaded LaTeX file.")

        generate_button = st.button(
            "Generate JD-Specific LaTeX Experience Section",
            type="primary",
        )

        if generate_button:
            if not portfolio_text or not latex_source:
                st.warning("Provide portfolio and LaTeX resume (via templates/ or upload).")
                return
            if not job_description.strip():
                st.warning("Please paste a job description in the sidebar.")
                return

            # Legacy single-shot prompt no longer used; role-by-role rewrite prevents cross-contamination.
            golden_prompt = ""

            # Show how many separate role calls we will make (typically 3: DSRS, Apptware ADS, Apptware Intern)
            current_experience = _extract_experience_section(latex_source) or ""
            blocks = _parse_experience_role_blocks(current_experience)
            if blocks:
                st.caption(
                    f"Rewriting **{len(blocks)}** experience blocks using **separate model calls per role** to prevent cross-contamination."
                )

            with st.spinner("Crafting a JD-specific Experience section from your portfolio and LaTeX..."):
                try:
                    if not current_experience:
                        st.error("Could not find an Experience section in the LaTeX resume.")
                        return

                    latex_experience, rationale_text = _golden_hand_from_text(
                        portfolio_text=portfolio_text,
                        latex_source=latex_source,
                        job_description=job_description,
                        model_name=model_name,
                    )
                except Exception as e:
                    st.error(f"Failed to generate LaTeX from LLM: {e}")
                    return

            if not latex_experience:
                st.error("The model did not return any LaTeX content. Try again or adjust the inputs.")
                return

            merged_tex = _merge_experience_into_tex(latex_source, latex_experience)
            pdf_bytes = _compile_tex_to_pdf(merged_tex)
            # Persist so switching pages does not lose output
            st.session_state["golden_hand_latex"] = latex_experience
            st.session_state["golden_hand_rationale"] = rationale_text
            st.session_state["golden_hand_rationale_parsed"] = _parse_golden_hand_rationale(rationale_text)
            st.session_state["golden_hand_merged_tex"] = merged_tex
            st.session_state["golden_hand_pdf_bytes"] = pdf_bytes
            # LLM-generated relevance metric (resume vs JD)
            resume_for_metric = ""
            if pdf_bytes:
                resume_for_metric = extract_text_from_pdf(BytesIO(pdf_bytes)) or ""
            if not resume_for_metric and merged_tex:
                resume_for_metric = _latex_to_plain_text(merged_tex)
            if resume_for_metric and job_description.strip():
                try:
                    rel_score, rel_expl = get_relevance_metric(
                        resume_for_metric, job_description, model_name
                    )
                    st.session_state["golden_hand_relevance_score"] = rel_score
                    st.session_state["golden_hand_relevance_explanation"] = rel_expl
                except Exception:
                    st.session_state["golden_hand_relevance_score"] = None
                    st.session_state["golden_hand_relevance_explanation"] = None
            else:
                st.session_state["golden_hand_relevance_score"] = None
                st.session_state["golden_hand_relevance_explanation"] = None

        # Show last Golden Hand result whenever we have it (so it survives page switches)
        if st.session_state.get("golden_hand_latex"):
            rationale_text = st.session_state.get("golden_hand_rationale", "")
            parsed = st.session_state.get("golden_hand_rationale_parsed") or _parse_golden_hand_rationale(rationale_text)
            latex_experience = st.session_state["golden_hand_latex"]
            pdf_bytes = st.session_state.get("golden_hand_pdf_bytes")

            # Compatibility Rating — only show when we have a real value (not "**" or empty)
            compat_raw = (parsed.get("compatibility_rating") or "").strip()
            compat = compat_raw.replace("*", "").strip()
            if compat and len(compat) >= 4 and re.search(r"\d", compat):
                st.markdown("#### Compatibility Rating")
                st.markdown(compat_raw)  # show original so "4.8 / 5.0 (High...)" is preserved
                st.markdown("---")

            # Rationale (why profile matches) — strip stray ** so they don't show as literal
            if parsed.get("rationale"):
                st.markdown("#### Rationale")
                st.markdown(_clean_rationale_display(parsed["rationale"]))
                st.markdown("---")

            # Relevance metric (resume vs JD)
            rel_score = st.session_state.get("golden_hand_relevance_score")
            rel_expl = st.session_state.get("golden_hand_relevance_explanation")
            if rel_score is not None:
                st.markdown("#### Relevance to job description")
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.metric("Relevance", f"{rel_score}/100")
                    st.progress(max(0, min(100, rel_score)) / 100)
                with c2:
                    st.caption(rel_expl or "")
                st.markdown("---")

            # Alterations: what changed and why (Original → Optimized → Why this improves it)
            if parsed.get("alterations"):
                with st.expander("#### Alterations (what changed and why)", expanded=True):
                    st.markdown(_clean_rationale_display(parsed["alterations"]))
                st.markdown("---")

            if rationale_text and not parsed.get("compatibility_rating") and not parsed.get("rationale"):
                st.markdown("#### Rationale & XYZ Structure")
                st.markdown(rationale_text)
                st.markdown("---")

            st.markdown("#### Updated LaTeX EXPERIENCE Section")
            st.caption(
                "Raw LaTeX only (no verbatim, no X/Y/Z labels). Paste into Overleaf where the Experience section lives."
            )
            # Strip leading/trailing ** so they are never copied
            latex_for_display = latex_experience.strip()
            if latex_for_display.startswith("**"):
                latex_for_display = latex_for_display[2:].lstrip()
            if latex_for_display.endswith("**"):
                latex_for_display = latex_for_display[:-2].rstrip()
            st.text_area(
                "LaTeX EXPERIENCE section",
                value=latex_for_display,
                height=260,
                key="golden_hand_latex_area",
            )
            col_copy, col_dl = st.columns([1, 1])
            with col_copy:
                # Copy to clipboard via HTML/JS (no ** in copied text)
                js_payload = json.dumps(latex_for_display)
                js_code = (
                    f"(function(){{"
                    f"var el=document.createElement('textarea');"
                    f"el.value={js_payload};"
                    f"document.body.appendChild(el);el.select();"
                    f"document.execCommand('copy');"
                    f"document.body.removeChild(el);"
                    f"alert('LaTeX copied to clipboard.');"
                    f"}})();"
                )
                js_escaped = (
                    js_code.replace("&", "&amp;")
                    .replace('"', "&quot;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                copy_html = f"""
                <button type="button" onclick="{js_escaped}" style="
                    padding: 0.4rem 0.8rem;
                    cursor: pointer;
                    border-radius: 4px;
                    border: 1px solid #ccc;
                    background: #f0f2f6;
                    font-size: 14px;
                " title="Copy full LaTeX">📋 Copy to clipboard</button>
                """
                st.markdown(copy_html, unsafe_allow_html=True)
            with col_dl:
                st.download_button(
                    "Download EXPERIENCE section as .tex snippet",
                    data=latex_for_display,
                    file_name="experience_the_golden_hand.tex",
                    mime="text/plain",
                    key="golden_hand_dl_tex",
                )

            st.markdown("---")
            st.markdown("#### Resume PDF")

            if pdf_bytes:
                b64 = base64.b64encode(pdf_bytes).decode()
                st.caption("Preview the compiled resume below. Download to save.")
                st.markdown(
                    f'<iframe src="data:application/pdf;base64,{b64}#toolbar=1" width="100%" height="600" type="application/pdf"></iframe>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "Download resume as PDF",
                    data=pdf_bytes,
                    file_name="resume_golden_hand.pdf",
                    mime="application/pdf",
                    key="golden_hand_dl_pdf",
                )
            else:
                st.info(
                    "PDF preview requires **pdflatex** (e.g. TeX Live or MacTeX). Paste the Experience section above "
                    "into your Overleaf resume to compile."
                )

            st.markdown("---")
            st.markdown("#### Roast this version")
            st.caption("Get a reality check and game plan for the resume you just generated (vs the job description).")
            roast_btn = st.button("Roast this version", type="secondary", key="golden_hand_roast_btn")
            if roast_btn and job_description.strip():
                merged_tex = st.session_state.get("golden_hand_merged_tex", "")
                resume_for_roast = ""
                if pdf_bytes:
                    resume_for_roast = extract_text_from_pdf(BytesIO(pdf_bytes)) or ""
                if not resume_for_roast and merged_tex:
                    resume_for_roast = _latex_to_plain_text(merged_tex)
                if not resume_for_roast:
                    st.warning("No resume text available to roast (compile PDF or use merged LaTeX).")
                else:
                    with st.spinner("Roasting your tailored resume..."):
                        try:
                            review = get_resume_review(
                                resume_text=resume_for_roast,
                                job_description=job_description,
                                roast_tone="Balanced",
                                model_name=model_name,
                            )
                            st.session_state["golden_hand_review"] = review
                        except Exception as e:
                            st.error(f"Roast failed: {e}")

            if st.session_state.get("golden_hand_review"):
                review = st.session_state["golden_hand_review"]
                st.markdown("##### The roast")
                st.error(review.roast)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Overall match", f"{review.overall_match_score}/100")
                    st.metric("Skills match", f"{review.skills_match_score}/100")
                    st.metric("Impact", f"{review.impact_score}/100")
                    if review.missing_keywords:
                        st.success("Keywords to strengthen")
                        st.markdown("\n".join(f"- **{kw}**" for kw in review.missing_keywords))
                with col_b:
                    st.markdown("##### Game plan")
                    st.info(review.coaching_notes)
                    if review.rewritten_bullets:
                        st.markdown("**Sample rewrites (XYZ)**")
                        for b in review.rewritten_bullets[:2]:
                            st.markdown(f"- {b.new}")

    elif page == "Job Scanner":
        st.markdown("### Job Scanner – Entry-Level New Grad Roles")
        st.caption(
            "This page reads from curated GitHub repos in `links/links.txt`, keeps only entry-level / new grad jobs "
            "and drops internships, and generates one Golden Hand-style resume tweak per job."
        )

        col_controls, col_info = st.columns([1, 2])
        with col_controls:
            if st.button("Run scan now (ingests all new jobs from links)", type="primary"):
                with st.spinner("Scanning curated repos for new entry-level roles..."):
                    try:
                        new_count, total = run_job_scan(model_name=model_name)
                        st.success(f"Scan complete. New jobs this run: {new_count}. Total stored: {total}.")
                    except Exception as e:
                        st.error(f"Job scan failed: {e}")
        with col_info:
            st.info(
                "An Airflow DAG (`airflow_dag.py`) can run this scan automatically every 6 hours. "
                "This button is a safe manual trigger for debugging or ad-hoc refreshes."
            )

        jobs = load_latest_results_mixed(limit=500)
        if not jobs:
            st.warning(
                "No jobs have been scanned yet. Ensure `links/links.txt` has the GitHub repos and that the scanner "
                "has run at least once (via Airflow or the button above)."
            )
        else:
            st.markdown("#### Latest Entry-Level / New Grad Jobs")
            sources = sorted({j.source_repo for j in jobs})
            st.caption(f"Showing {len(jobs)} jobs from: {', '.join(sources)}. Sorted newest first (Age / Date Posted).")

            # History: collapsible list of previously loaded JDs (title + JD per sub-expander)
            jd_cache = st.session_state.get("job_jd_cache", {})
            if jd_cache:
                with st.expander("History (loaded JDs)", expanded=False):
                    for hi, (hist_key, hist_data) in enumerate(list(jd_cache.items())):
                        if not isinstance(hist_data, dict):
                            continue
                        # Skip invalid entries (jobright.ai site link, not a job)
                        if "jobright.ai|jobright.ai" in hist_key or hist_data.get("title") == "jobright.ai":
                            continue
                        # Prefer list title (job title from repo) so History shows correct job name
                        ht = hist_data.get("title") or hist_data.get("jd_title") or (hist_data.get("jd_text") or "")[:60] or hist_key
                        if isinstance(ht, str) and len(ht) > 80:
                            ht = ht[:77] + "..."
                        with st.expander(ht or "Job", expanded=False):
                            st.markdown("**Title:** " + (hist_data.get("title") or hist_data.get("jd_title") or "—"))
                            loc = hist_data.get("location")
                            age = hist_data.get("age_days")
                            dpost = hist_data.get("date_posted")
                            sal = hist_data.get("salary")
                            if loc or age is not None or dpost or sal:
                                meta = []
                                if loc:
                                    meta.append(f"Location: {loc}")
                                if sal:
                                    meta.append(f"Salary: {sal}")
                                if age is not None:
                                    meta.append(f"Age: {age}d")
                                if dpost:
                                    meta.append(f"Date posted: {dpost}")
                                st.caption(" · ".join(meta))
                            jd_hist = (hist_data.get("jd_text") or "")[:12000]
                            if len(hist_data.get("jd_text") or "") > 12000:
                                jd_hist += "\n...[truncated]..."
                            st.text_area("Job description", value=jd_hist, height=200, key=f"hist_{hi}", disabled=True)

            # Prepare shared resources for per-job Golden Hand + cover letter
            tc = st.session_state["template_checks"]
            portfolio_text = ""
            latex_source = ""
            if tc["portfolio"].is_file():
                portfolio_text = extract_text_from_pdf(BytesIO(tc["portfolio"].read_bytes()))
            if tc["resume_tex"].is_file():
                latex_source = tc["resume_tex"].read_text(encoding="utf-8", errors="ignore")

            if "job_scanner_results" not in st.session_state:
                st.session_state["job_scanner_results"] = {}
            if "job_jd_cache" not in st.session_state:
                st.session_state["job_jd_cache"] = {}

            results = st.session_state["job_scanner_results"]
            jd_cache = st.session_state["job_jd_cache"]

            for idx, job in enumerate(jobs, start=1):
                expander_label = f"{idx}. {job.title or 'Job posting'}"
                with st.expander(expander_label, expanded=False):
                    title_line = "**Job title (from list):** " + (job.title or "—")
                    if getattr(job, "is_faang", False):
                        title_line += ' <span style="display:inline-block;background:#22c55e;color:#fff;font-size:0.75rem;font-weight:600;padding:2px 8px;border-radius:4px;margin-left:6px;">FAANG</span>'
                    st.markdown(title_line, unsafe_allow_html=True)
                    job_url = get_job_posting_url(job)
                    st.markdown(f"[Open job posting]({job_url})")
                    age_days = getattr(job, "age_days", None)
                    date_posted = getattr(job, "date_posted", None)
                    location = getattr(job, "location", None)
                    salary = getattr(job, "salary", None)
                    cap_parts = [f"Source: `{job.source_repo}`"]
                    if location:
                        cap_parts.append(f"Location: {location}")
                    if salary:
                        cap_parts.append(f"Salary: {salary}")
                    if age_days is not None:
                        cap_parts.append(f"Age: {age_days}d")
                    elif date_posted:
                        cap_parts.append(f"Date posted: {date_posted}")
                    st.caption(" · ".join(cap_parts))

                    job_key = job.id
                    cached_jd = jd_cache.get(job_key)

                    if cached_jd:
                        posting_title = cached_jd.get("jd_title") or job.title or "—"
                        st.markdown("**Title from posting:** " + posting_title)
                        jd_snippet = (cached_jd.get("jd_text") or "")[:6000]
                        if len(cached_jd.get("jd_text") or "") > 6000:
                            jd_snippet += "\n...[truncated]..."
                        st.text_area(
                            "Job description (from posting)",
                            value=jd_snippet,
                            height=220,
                            key=f"job_scanner_jd_display_{idx}",
                            disabled=True,
                        )
                    else:
                        if st.button("Load job description", key=f"job_load_jd_{idx}"):
                            with st.spinner("Fetching job description..."):
                                jd_text, jd_title = _fetch_jd(job_url)
                                if jd_text or jd_title:
                                    jd_cache[job_key] = {
                                        "jd_text": jd_text or "",
                                        "jd_title": jd_title or "",
                                        "title": job.title,
                                        "location": getattr(job, "location", None),
                                        "age_days": getattr(job, "age_days", None),
                                        "date_posted": getattr(job, "date_posted", None),
                                        "salary": getattr(job, "salary", None),
                                    }
                                    st.rerun()
                                else:
                                    st.error("Could not fetch this job posting. The link may be a company homepage; try opening it in a browser.")

                    if st.button(
                        "Generate Golden Hand LaTeX + Cover Letter for this job",
                        key=f"job_scanner_btn_{idx}",
                    ):
                        if not portfolio_text or not latex_source:
                            st.error(
                                "Golden Hand requires `templates/portfolio.pdf` and `templates/resume.tex`. "
                                "Please add them and try again."
                            )
                        else:
                            if cached_jd and cached_jd.get("jd_text"):
                                jd_text, jd_title = cached_jd.get("jd_text", ""), cached_jd.get("jd_title", "")
                            else:
                                jd_text, jd_title = _fetch_jd(job_url)
                            if not jd_text:
                                st.error("Could not fetch or parse the job description for this posting.")
                            else:
                                with st.spinner("Running Golden Hand and generating cover letter for this job..."):
                                    try:
                                        latex_experience, rationale_text = _golden_hand_from_text(
                                            portfolio_text=portfolio_text,
                                            latex_source=latex_source,
                                            job_description=jd_text,
                                            model_name=model_name,
                                        )
                                    except Exception as e:
                                        st.error(f"Golden Hand failed: {e}")
                                        latex_experience = ""
                                        rationale_text = ""

                                    rationale_parsed = _parse_golden_hand_rationale(rationale_text) if rationale_text else {}
                                    cover_letter_text = ""
                                    if latex_experience:
                                        # Use resume.tex (roughly stripped) for cover letter context
                                        resume_plain = _latex_to_plain_text(latex_source)
                                        template_text, _ = _load_cover_letter_template()
                                        if template_text:
                                            try:
                                                cover_letter_text = get_cover_letter(
                                                    resume_text=resume_plain,
                                                    job_description=jd_text,
                                                    template=template_text,
                                                    model_name=model_name,
                                                )
                                            except Exception as e:
                                                st.error(f"Cover letter generation failed: {e}")

                                    results[job_key] = {
                                        "latex": latex_experience,
                                        "cover_letter": cover_letter_text,
                                        "jd_text": jd_text,
                                        "jd_title": jd_title,
                                        "rationale_parsed": rationale_parsed,
                                    }
                                    jd_cache[job_key] = {
                                        "jd_text": jd_text,
                                        "jd_title": jd_title or "",
                                        "title": job.title,
                                        "location": getattr(job, "location", None),
                                        "age_days": getattr(job, "age_days", None),
                                        "date_posted": getattr(job, "date_posted", None),
                                        "salary": getattr(job, "salary", None),
                                    }

                    job_result = results.get(job_key)
                    if job_result and job_result.get("latex"):
                        rp = job_result.get("rationale_parsed") or {}
                        compat_raw = (rp.get("compatibility_rating") or "").strip()
                        compat_clean = compat_raw.replace("*", "").strip()
                        if compat_clean and len(compat_clean) >= 4 and re.search(r"\d", compat_clean):
                            st.markdown("**Compatibility Rating:** " + compat_raw)
                        if rp.get("rationale"):
                            with st.expander("Rationale (why your profile matches)", expanded=False):
                                st.markdown(_clean_rationale_display(rp["rationale"]))
                        if rp.get("alterations"):
                            with st.expander("Alterations (what changed and why)", expanded=True):
                                st.markdown(_clean_rationale_display(rp["alterations"]))
                        st.markdown("**LaTeX Experience section**")
                        latex_for_display = job_result["latex"].strip()
                        st.text_area(
                            "LaTeX EXPERIENCE section",
                            value=latex_for_display,
                            height=220,
                            key=f"job_scanner_latex_area_{idx}",
                        )
                        c1, c2 = st.columns(2)
                        with c1:
                            js_payload = json.dumps(latex_for_display)
                            js_code = (
                                f"(function(){{"
                                f"var el=document.createElement('textarea');"
                                f"el.value={js_payload};"
                                f"document.body.appendChild(el);el.select();"
                                f"document.execCommand('copy');"
                                f"document.body.removeChild(el);"
                                f"alert('LaTeX copied to clipboard.');"
                                f"}})();"
                            )
                            js_escaped = (
                                js_code.replace("&", "&amp;")
                                .replace('"', "&quot;")
                                .replace("<", "&lt;")
                                .replace(">", "&gt;")
                            )
                            copy_html = f"""
                            <button type="button" onclick="{js_escaped}" style="
                                padding: 0.4rem 0.8rem;
                                cursor: pointer;
                                border-radius: 4px;
                                border: 1px solid #ccc;
                                background: #f0f2f6;
                                font-size: 14px;
                            " title="Copy full LaTeX">📋 Copy to clipboard</button>
                            """
                            st.markdown(copy_html, unsafe_allow_html=True)
                        with c2:
                            st.download_button(
                                "Download .tex",
                                data=latex_for_display,
                                file_name=f"experience_job_{idx}.tex",
                                mime="text/plain",
                                key=f"job_scanner_dl_tex_{idx}",
                            )

                    if job_result and job_result.get("cover_letter"):
                        st.markdown("**Cover letter**")
                        cl_text = job_result["cover_letter"]
                        st.text_area(
                            "Cover letter",
                            value=cl_text,
                            height=220,
                            key=f"job_scanner_cover_letter_area_{idx}",
                        )
                        st.download_button(
                            "Download cover letter",
                            data=cl_text,
                            file_name=f"cover_letter_job_{idx}.txt",
                            mime="text/plain",
                            key=f"job_scanner_cover_letter_dl_{idx}",
                        )

    elif page == "Cover Letter":
        tc = st.session_state["template_checks"]
        cover_path = tc["cover_letter"]
        if not cover_path.is_file():
            st.error(
                f"Cover letter template not found at `{cover_path}`. Create `templates/cover_letter.txt` in the project "
                "folder (or set TEMPLATES_DIR / PROJECT_ROOT)."
            )
        else:
            cover_template = cover_path.read_text(encoding="utf-8").strip()
            if not cover_template:
                st.error("`templates/cover_letter.txt` is empty.")
            else:
                st.markdown("### Cover Letter Generator")
                st.caption(f"Using template: `{cover_path}`")
                # Prefer resume from templates (resume.pdf or portfolio.pdf) so user doesn't re-upload
                resume_text_cl = ""
                if tc["resume_pdf"].is_file():
                    resume_text_cl = extract_text_from_pdf(BytesIO(tc["resume_pdf"].read_bytes()))
                    st.success(f"Using **templates/resume.pdf** as resume for the letter.")
                elif tc["portfolio"].is_file():
                    resume_text_cl = extract_text_from_pdf(BytesIO(tc["portfolio"].read_bytes()))
                    st.success(f"Using **templates/portfolio.pdf** as resume for the letter.")
                resume_pdf_cl = st.file_uploader(
                    "Override resume (optional if using templates)",
                    type=["pdf"],
                    key="cover_letter_resume",
                )
                if resume_pdf_cl:
                    resume_text_cl = extract_text_from_pdf(resume_pdf_cl) or resume_text_cl

                # Clear cached cover letter when sidebar JD changes so we never show a letter for the wrong company
                jd_used = st.session_state.get("jd_used_for_cover_letter", "")
                current_jd = job_description.strip()
                # Compare first 2000 chars so we clear when user pastes a different JD (e.g. Extend vs Root)
                jd_match = (current_jd[:2000] == jd_used[:2000]) if (jd_used and current_jd) else False
                if st.session_state.get("cover_letter_text") and not jd_match:
                    st.session_state.pop("cover_letter_text", None)
                    st.session_state.pop("jd_used_for_cover_letter", None)

                gen_cover = st.button("Generate Cover Letter", type="primary", key="cover_letter_btn")
                if gen_cover:
                    if not resume_text_cl:
                        st.warning("Provide a resume via templates (resume.pdf or portfolio.pdf) or upload.")
                    elif not job_description.strip():
                        st.warning("Please paste the job description in the sidebar.")
                    else:
                        with st.spinner("Generating cover letter..."):
                            try:
                                letter = get_cover_letter(
                                    resume_text=resume_text_cl,
                                    job_description=job_description,
                                    template=cover_template,
                                    model_name=model_name,
                                )
                                st.session_state["cover_letter_text"] = letter
                                st.session_state["jd_used_for_cover_letter"] = job_description
                            except Exception as e:
                                st.error(f"Failed to generate cover letter: {e}")

                if st.session_state.get("cover_letter_text"):
                    st.markdown("#### Your Cover Letter")
                    cl_text = st.session_state["cover_letter_text"]
                    st.text_area("Cover letter", value=cl_text, height=400, key="cover_letter_area")
                    st.download_button(
                        "Download as .txt",
                        data=cl_text,
                        file_name="cover_letter.txt",
                        mime="text/plain",
                        key="cover_letter_dl",
                    )


if __name__ == "__main__":
    main()

