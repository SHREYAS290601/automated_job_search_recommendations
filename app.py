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

from jobs_pipeline import load_latest_results, run_job_scan


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
    "specific job. You are given: (1) the candidate's resume text, (2) the job description, and (3) a sample cover "
    "letter template. Generate a new cover letter that: preserves the structure, tone, and paragraph flow of the "
    "template; fills in the correct company name, role title, and key requirements from the job description; and "
    "weaves in concrete achievements and skills from the resume that match the JD. Write in first person as the "
    "candidate. Output only the cover letter text—no headings, no meta-commentary."
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


def get_cover_letter(resume_text: str, job_description: str, template: str, model_name: str) -> str:
    """Generate a tailored cover letter from resume, JD, and template."""
    if not template.strip():
        raise ValueError("Cover letter template is empty.")
    user_content = (
        "Generate a tailored cover letter using the following inputs.\n\n"
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


def _fetch_jd_text(url: str) -> str | None:
    """Fetch a job description page and return plain text, or None on failure."""
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n")
        return text.strip() or None
    except Exception:
        return None


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
    prompt = (
        "Two resume bullets for a Data Science Intern role. Write ONE more in the same XYZ style. "
        "One line only. No \\item, no LaTeX.\n\nBullets:\n" + two_text + "\n\nJD:\n" + job_description[:1000]
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
    golden_prompt = (
        "You are helping a candidate maintain a single LaTeX resume source file while tailoring it "
        "to a specific job description.\n\n"
        "You are given:\n"
        "1) A portfolio PDF (parsed text) containing ALL of their experiences and achievements.\n"
        "2) Their current LaTeX resume source.\n"
        "3) A target job description.\n\n"
        "Your job:\n"
        "- Focus ONLY on the EXPERIENCE section of the LaTeX resume.\n"
        "- Using the portfolio text and job description, REWRITE and TAILOR bullets (XYZ format, JD keywords) "
        "but do NOT remove any content. The output must include every role and every bullet from the current "
        "resume/portfolio—nothing may be dropped even if it seems less relevant to the job description.\n"
        "- Preserve LaTeX structure, macros, environments, and formatting style.\n"
        "- Do NOT invent new companies, titles, or technologies that do not appear in the portfolio text.\n"
        "- You may reorder bullets within a role for emphasis and rewrite them in XYZ form; you may add 1–2 "
        "JD-focused bullets per role if the portfolio supports it. Do NOT remove or omit any existing role, "
        "company, date range, or bullet—the PDF must not lose any line that appears in the source.\n"
        "- ONE-PAGE TARGET: Two-page resumes are not acceptable. Use \\small or \\footnotesize at the start of "
        "the Experience section (e.g. after the section heading) so the resume fits on one page. Use strict, "
        "consistent spacing: no extra spaces inside company names (e.g. 'Apptware Pvt. Ltd., Pune' with single "
        "spaces only); do not add unintended small spaces.\n"
        "- DATA SCIENCE INTERN (or the earliest/most junior role) must have exactly 3 bullet points—no more, "
        "no fewer. Choose the 3 most relevant to the JD from the portfolio. All other roles keep their bullets "
        "as in the source (or as tailored).\n"
        "- LaTeX structure (strict): Use \\noindent immediately after the section heading, before "
        "\\rule{\\textwidth}{0.4pt}, and before each company/role block (before each \\textbf{Company...} line). "
        "Follow the exact structure of the current resume: \\section*{...} then \\noindent, \\rule{...}, \\small, "
        "then for each role \\noindent before \\textbf{...}. Do not omit \\noindent.\n"
        "- Do NOT include any LaTeX comments or meta-text in the output. Never write '% added JD-focused bullet' "
        "or any comment—only content that should appear on the printed resume.\n"
        "- All bullets must follow the XYZ resume formula (Google-style): "
        "Accomplished [outcome] as measured by [metric], by doing [action]. "
        "Write natural sentences that convey this structure. Do NOT include literal (X), (Y), (Z) or [X], [Y], [Z] "
        "labels in the bullet text—they must read as professional prose only.\n"
        "- Output ONLY raw LaTeX. Do NOT wrap the experience section in \\begin{verbatim} or \\begin{lstlisting} "
        "or any code environment. The LaTeX must be pasteable directly into a resume document.\n\n"
        "OUTPUT FORMAT (STRICT):\n"
        "You must respond in three clearly labelled sections, in this exact order:\n"
        "RATIONALE:\n"
        "- 1–2 short paragraphs explaining the reasoning behind the changes and how the XYZ structure was applied.\n\n"
        "XYZ_FORMAT_EXAMPLES:\n"
        "- 2–3 plain-text example bullets (not LaTeX) that show the XYZ structure, without (X)/(Y)/(Z) labels.\n\n"
        "LATEX_EXPERIENCE:\n"
        "- ONLY the UPDATED LaTeX code for the EXPERIENCE section (e.g. \\section*{...} through the end of that "
        "section). Raw LaTeX only—no verbatim, no markdown, no commentary.\n"
        "- The section must be COMPLETE: include every role; Data Science Intern (or earliest role) has exactly "
        "3 bullets; other roles keep all their bullets. Do not omit \\noindent before each role block.\n"
        "- In LaTeX, percent signs must be written as \\% (e.g. 50\\%) or the rest of the line will be truncated.\n"
        "- Ensure the section is the ENTIRE Experience section only (all roles, all bullets as above); no other "
        "sections. Output only this section.\n\n"
        "=== PORTFOLIO TEXT ===\n"
        f"{portfolio_text}\n\n"
        "=== CURRENT LATEX RESUME ===\n"
        f"{latex_source}\n\n"
        "=== JOB DESCRIPTION ===\n"
        f"{job_description}\n"
    )

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": golden_prompt,
            }
        ],
    )
    raw_text = _extract_text_from_response(response).strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:tex|latex)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```\s*$", "", raw_text)

    rationale_text = ""
    latex_experience = raw_text
    if "LATEX_EXPERIENCE:" in raw_text:
        before, after = raw_text.split("LATEX_EXPERIENCE:", 1)
        rationale_text = before.strip()
        latex_experience = after.strip()

    if not latex_experience:
        raise ValueError("Golden Hand returned no LaTeX experience section.")

    latex_experience = _clean_latex_experience(latex_experience)
    latex_experience = _ensure_noindent_before_roles(latex_experience)
    latex_experience = _ensure_data_science_intern_three_bullets(
        latex_experience,
        portfolio_text=portfolio_text,
        job_description=job_description,
        model_name=model_name,
    )
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

            golden_prompt = (
                "You are helping a candidate maintain a single LaTeX resume source file while tailoring it "
                "to a specific job description.\n\n"
                "You are given:\n"
                "1) A portfolio PDF (parsed text) containing ALL of their experiences and achievements.\n"
                "2) Their current LaTeX resume source.\n"
                "3) A target job description.\n\n"
                "Your job:\n"
                "- Focus ONLY on the EXPERIENCE section of the LaTeX resume.\n"
                "- Using the portfolio text and job description, REWRITE and TAILOR bullets (XYZ format, JD keywords) "
                "but do NOT remove any content. The output must include every role and every bullet from the current "
                "resume/portfolio—nothing may be dropped even if it seems less relevant to the job description.\n"
                "- Preserve LaTeX structure, macros, environments, and formatting style.\n"
                "- Do NOT invent new companies, titles, or technologies that do not appear in the portfolio text.\n"
                "- You may reorder bullets within a role for emphasis and rewrite them in XYZ form; you may add 1–2 "
                "JD-focused bullets per role if the portfolio supports it. Do NOT remove or omit any existing role, "
                "company, date range, or bullet—the PDF must not lose any line that appears in the source.\n"
                "- ONE-PAGE TARGET: Two-page resumes are not acceptable. Use \\small or \\footnotesize at the start of "
                "the Experience section (e.g. after the section heading) so the resume fits on one page. Use strict, "
                "consistent spacing: no extra spaces inside company names (e.g. 'Apptware Pvt. Ltd., Pune' with single "
                "spaces only); do not add unintended small spaces.\n"
                "- DATA SCIENCE INTERN (or the earliest/most junior role) must have exactly 3 bullet points—no more, "
                "no fewer. Choose the 3 most relevant to the JD from the portfolio. All other roles keep their bullets "
                "as in the source (or as tailored).\n"
                "- LaTeX structure (strict): Use \\noindent immediately after the section heading, before "
                "\\rule{\\textwidth}{0.4pt}, and before each company/role block (before each \\textbf{Company...} line). "
                "Follow the exact structure of the current resume: \\section*{...} then \\noindent, \\rule{...}, \\small, "
                "then for each role \\noindent before \\textbf{...}. Do not omit \\noindent.\n"
                "- Do NOT include any LaTeX comments or meta-text in the output. Never write '% added JD-focused bullet' "
                "or any comment—only content that should appear on the printed resume.\n"
                "- All bullets must follow the XYZ resume formula (Google-style): "
                "Accomplished [outcome] as measured by [metric], by doing [action]. "
                "Write natural sentences that convey this structure. Do NOT include literal (X), (Y), (Z) or [X], [Y], [Z] "
                "labels in the bullet text—they must read as professional prose only.\n"
                "- Output ONLY raw LaTeX. Do NOT wrap the experience section in \\begin{verbatim} or \\begin{lstlisting} "
                "or any code environment. The LaTeX must be pasteable directly into a resume document.\n\n"
                "OUTPUT FORMAT (STRICT):\n"
                "You must respond in three clearly labelled sections, in this exact order:\n"
                "RATIONALE:\n"
                "- 1–2 short paragraphs explaining the reasoning behind the changes and how the XYZ structure was applied.\n\n"
                "XYZ_FORMAT_EXAMPLES:\n"
                "- 2–3 plain-text example bullets (not LaTeX) that show the XYZ structure, without (X)/(Y)/(Z) labels.\n\n"
                "LATEX_EXPERIENCE:\n"
                "- ONLY the UPDATED LaTeX code for the EXPERIENCE section (e.g. \\section*{...} through the end of that "
                "section). Raw LaTeX only—no verbatim, no markdown, no commentary.\n"
                "- The section must be COMPLETE: include every role; Data Science Intern (or earliest role) has exactly "
                "3 bullets; other roles keep all their bullets. Do not omit \\noindent before each role block.\n"
                "- In LaTeX, percent signs must be written as \\% (e.g. 50\\%) or the rest of the line will be truncated.\n"
                "- Ensure the section is the ENTIRE Experience section only (all roles, all bullets as above); no other "
                "sections. Output only this section.\n\n"
                "=== PORTFOLIO TEXT ===\n"
                f"{portfolio_text}\n\n"
                "=== CURRENT LATEX RESUME ===\n"
                f"{latex_source}\n\n"
                "=== JOB DESCRIPTION ===\n"
                f"{job_description}\n"
            )

            with st.spinner("Crafting a JD-specific Experience section from your portfolio and LaTeX..."):
                try:
                    response = client.responses.create(
                        model=model_name,
                        input=[
                            {
                                "role": "user",
                                "content": golden_prompt,
                            }
                        ],
                    )
                    raw_text = _extract_text_from_response(response).strip()
                    if raw_text.startswith("```"):
                        raw_text = re.sub(r"^```(?:tex|latex)?\\s*", "", raw_text)
                        raw_text = re.sub(r"\\s*```\\s*$", "", raw_text)

                    rationale_text = ""
                    latex_experience = raw_text

                    # Split out the LATEX_EXPERIENCE block if present.
                    if "LATEX_EXPERIENCE:" in raw_text:
                        before, after = raw_text.split("LATEX_EXPERIENCE:", 1)
                        rationale_text = before.strip()
                        latex_experience = after.strip()
                except Exception as e:
                    st.error(f"Failed to generate LaTeX from LLM: {e}")
                    return

            if not latex_experience:
                st.error("The model did not return any LaTeX content. Try again or adjust the inputs.")
                return

            # Clean: remove verbatim wrapper and (X)/(Y)/(Z) labels
            latex_experience = _clean_latex_experience(latex_experience)
            # Enforce \noindent before each role and exactly 3 bullets for Data Science Intern
            latex_experience = _ensure_noindent_before_roles(latex_experience)
            latex_experience = _ensure_data_science_intern_three_bullets(
                latex_experience, portfolio_text, job_description, model_name
            )
            merged_tex = _merge_experience_into_tex(latex_source, latex_experience)
            pdf_bytes = _compile_tex_to_pdf(merged_tex)
            # Persist so switching pages does not lose output
            st.session_state["golden_hand_latex"] = latex_experience
            st.session_state["golden_hand_rationale"] = rationale_text
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
            latex_experience = st.session_state["golden_hand_latex"]
            pdf_bytes = st.session_state.get("golden_hand_pdf_bytes")

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

            if rationale_text:
                st.markdown("#### Rationale & XYZ Structure")
                st.markdown(rationale_text)

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
            if st.button("Run scan now (max 10 new jobs)", type="primary"):
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

        jobs = load_latest_results(limit=10)
        if not jobs:
            st.warning(
                "No jobs have been scanned yet. Ensure `links/links.txt` has the GitHub repos and that the scanner "
                "has run at least once (via Airflow or the button above)."
            )
        else:
            st.markdown("#### Latest Entry-Level / New Grad Jobs")

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

            results = st.session_state["job_scanner_results"]

            for idx, job in enumerate(jobs, start=1):
                with st.expander(f"{idx}. {job.title}", expanded=False):
                    st.markdown(f"[Open job posting]({job.url})")
                    st.caption(f"Source: `{job.source_repo}`")

                    job_key = job.id
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
                            jd_text = _fetch_jd_text(job.url)
                            if not jd_text:
                                st.error("Could not fetch or parse the job description for this posting.")
                            else:
                                with st.spinner("Running Golden Hand and generating cover letter for this job..."):
                                    try:
                                        latex_experience, _ = _golden_hand_from_text(
                                            portfolio_text=portfolio_text,
                                            latex_source=latex_source,
                                            job_description=jd_text,
                                            model_name=model_name,
                                        )
                                    except Exception as e:
                                        st.error(f"Golden Hand failed: {e}")
                                        latex_experience = ""

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
                                    }

                    job_result = results.get(job_key)
                    if job_result and job_result.get("latex"):
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

