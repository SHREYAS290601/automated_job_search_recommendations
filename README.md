# Reality Check ATS 🛑

AI-powered resume tailoring and job scanner for New Grad roles, built with **Streamlit**, **PyMuPDF**, and a custom LLM endpoint.

You:
- Upload your resume (LaTeX + portfolio PDF).
- Paste or fetch job descriptions.
- Get a brutally honest roast, a fully rewritten **LaTeX Experience section (Golden Hand)**, and a tailored cover letter.
- See up-to-date **entry-level / new grad jobs** from curated GitHub lists, and generate per-job LaTeX + cover letters on demand.

---

## Features

### 1. The Golden Hand

- Reads your **portfolio PDF** and **LaTeX resume (`templates/portfolio.pdf`, `templates/resume.tex`)**.
- Takes a job description (JD) from the sidebar.
- Produces:
  - A full LaTeX `\section*{Professional Experience}`:
    - Enforces XYZ structure.
    - Uses `\noindent`, `\small`, consistent spacing.
    - Forces **exactly 3 bullets** for the Data Science Intern role.
  - Roast + match scores and coaching notes.
  - A compiled PDF preview (if `pdflatex` is installed locally).

### 2. Job Scanner

- Fetches and filters **entry-level / new grad** jobs from curated GitHub repos listed in `links/links.txt`.
- Drops internships / co-ops by keyword.
- Shows **up to 10 recent jobs** as collapsible rows.
- For each job you can:
  - Fetch its JD from the job URL.
  - Run a **Golden Hand** pipeline specialized to that JD to generate a **per-job LaTeX Experience section**.
  - Generate a **per-job cover letter** from:
    - Your stripped LaTeX resume.
    - The JD text.
    - `templates/cover_letter.txt` as a style/template.

### 3. Cover Letter

- Dedicated page that:
  - Loads `templates/cover_letter.txt` as a base template.
  - Takes your resume (from `templates/resume.pdf` or `templates/portfolio.pdf`) and a JD.
  - Produces a tailored cover letter ready to paste into applications.

---

## Project Structure (key files)

- `app.py` – Streamlit UI and Golden Hand / Job Scanner / Cover Letter logic.
- `jobs_pipeline.py` – GitHub markdown scanner:
  - Reads `links/links.txt`.
  - Filters to entry-level / new grad roles.
  - Writes `data/seen_jobs.json` and `data/job_results.json`.
- `links/links.txt` – List of GitHub repos that maintain New Grad job markdown lists.
- `templates/`
  - `portfolio.pdf` – your full experience portfolio.
  - `resume.tex` – your canonical LaTeX resume.
  - `resume.pdf` (optional) – used by Cover Letter where present.
  - `cover_letter.txt` – base cover letter template.
- `data/`
  - `seen_jobs.json` – IDs of jobs already processed.
  - `job_results.json` – most recent list of parsed jobs (consumed by Job Scanner).
- `.github/workflows/job_scan.yml` – GitHub Actions workflow that runs the scanner every 6 hours.
- `run_app.sh` – Local convenience script:
  - Runs tests.
  - Starts Airflow (for local experimentation) in Docker.
  - Launches `streamlit run app.py`.

---

## Local Development

### 1. Setup

```bash
git clone <your-repo-url>
cd vibe_code_project_roast_resume

python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://your-custom-base-url/v1  # or leave blank for direct OpenAI
```

Place your templates:

- `templates/portfolio.pdf`
- `templates/resume.tex`
- `templates/resume.pdf` (optional)
- `templates/cover_letter.txt`

### 2. Run tests

```bash
pytest
```

### 3. Run the app locally

You can run the Streamlit app directly:

```bash
streamlit run app.py
```

Or use the convenience script (runs tests first and starts local Airflow via Docker for experimentation):

```bash
chmod +x run_app.sh
./run_app.sh
```

> Note: Airflow is **not required** for production or deployment; it’s only used locally here. In production we use GitHub Actions to keep job data fresh.

---

## Automated Job Scanning with GitHub Actions

The workflow `.github/workflows/job_scan.yml`:

- Runs every 6 hours (and supports manual triggers).
- Installs dependencies with `pip install -r requirements.txt`.
- Executes:

```bash
python -c "from jobs_pipeline import run_job_scan; run_job_scan()"
```

- Commits updated `data/seen_jobs.json` and `data/job_results.json` back to the repo.

### GitHub setup

1. Push this repo to GitHub.
2. In the repo:
   - Go to **Settings → Secrets and variables → Actions**.
   - Add:
     - `OPENAI_API_KEY`
     - `OPENAI_BASE_URL` (if using a custom endpoint).
3. In the **Actions** tab:
   - Enable the **“Scan New Grad Jobs”** workflow.
   - Optionally run it manually to seed job data.

Your Streamlit app reads `data/job_results.json` directly, so it always sees the latest scanned jobs.

---

## Deploying on Streamlit Community Cloud (free)

1. Push the repo (with `app.py`, `requirements.txt`, `.github/workflows/job_scan.yml`, `templates/`, `links/`, `data/`) to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io) and create a new app:
   - Select your repo and branch.
   - Entry point: `app.py`.
3. In the Streamlit app’s settings:
   - Add environment variables:
     - `OPENAI_API_KEY`
     - `OPENAI_BASE_URL`
4. The live app will:
   - Serve **The Golden Hand**, **Job Scanner**, and **Cover Letter** pages.
   - Show New Grad roles from `data/job_results.json`, updated automatically by the GitHub Actions workflow.

---

## Notes & Guardrails

- Job Scanner:
  - Only keeps **entry-level / new grad** jobs.
  - Filters out internship / co-op by keywords.
  - Limits to 10 new jobs per scan.
- Golden Hand:
  - Never invents companies or technologies not present in your portfolio.
  - Enforces strict LaTeX formatting and bullet structure.
- Per-job generation:
  - Golden Hand LaTeX and cover letters are generated **on demand per job** to control latency and cost.

