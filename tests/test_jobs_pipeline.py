import json
from pathlib import Path

import pytest

import jobs_pipeline as jp


def test_github_url_to_raw_repo_root():
    url = "https://github.com/jobright-ai/2026-Data-Analysis-New-Grad"
    raw = jp._github_url_to_raw(url)
    assert raw == (
        "https://raw.githubusercontent.com/jobright-ai/2026-Data-Analysis-New-Grad/HEAD/README.md"
    )


def test_github_url_to_raw_blob():
    url = "https://github.com/speedyapply/2026-AI-College-Jobs/blob/main/NEW_GRAD_USA.md"
    raw = jp._github_url_to_raw(url)
    assert raw == (
        "https://raw.githubusercontent.com/speedyapply/2026-AI-College-Jobs/main/NEW_GRAD_USA.md"
    )


def test_parse_jobright_table():
    md = """
| Company | Job Title | Location | Work Model | Date Posted |
| ----- | --------- |  --------- | ---- | ------- |
| **[Indiana University](https://www.iu.edu)** | **[Research Data Assistant](https://jobright.ai/jobs/info/abc123)** | Indianapolis, IN | On Site | Mar 11 |
"""
    jobs = jp._parse_jobright_table(md, source_repo="2026-Data-Analysis-New-Grad")
    assert len(jobs) == 1
    assert jobs[0].title == "Indiana University - Research Data Assistant"
    assert jobs[0].url == "https://jobright.ai/jobs/info/abc123"
    assert jobs[0].is_faang is False


def test_parse_speedyapply_faang_tag():
    md = """
<!-- TABLE_FAANG_START -->
| Company | Position | Location | Salary | Posting | Age |
| <a href="https://www.nvidia.com"><strong>NVIDIA</strong></a> | Deep Learning Architect | Santa Clara | $172k/yr | <a href="https://nvidia.wd5.myworkdayjobs.com/apply"><img src="https://i.imgur.com/x.png" alt="Apply" width="70"/></a> | 1d |
<!-- TABLE_FAANG_END -->
"""
    jobs = jp._parse_speedyapply_tables(md, source_repo="NEW_GRAD_USA")
    assert len(jobs) == 1
    assert "NVIDIA" in jobs[0].title
    assert jobs[0].url == "https://nvidia.wd5.myworkdayjobs.com/apply"
    assert jobs[0].is_faang is True


def test_run_job_scan_uses_seen_state_and_limits(tmp_path, monkeypatch):
    """
    End-to-end-ish test of run_job_scan with:
    - Temporary data directory
    - Fake links.txt
    - Fake markdown fetch
    - Stubbed LLM suggestion generator
    """
    # Redirect DATA_DIR and related files to temp directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(jp, "DATA_DIR", data_dir, raising=False)
    monkeypatch.setattr(jp, "STATE_FILE", data_dir / "seen_jobs.json", raising=False)
    monkeypatch.setattr(jp, "RESULTS_FILE", data_dir / "job_results.json", raising=False)

    # Create fake links.txt pointing to two repos (one intern, one new-grad)
    links_dir = tmp_path / "links"
    links_dir.mkdir()
    links_file = links_dir / "links.txt"
    links_file.write_text(
        "https://github.com/jobright-ai/2026-Data-Analysis-New-Grad\n"
        "https://github.com/SimplifyJobs/Summer2026-Internships\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(jp, "LINKS_FILE", links_file, raising=False)

    # Fake markdown content: jobright table format
    def fake_fetch_markdown(url: str, timeout: int = 10) -> str | None:  # type: ignore[override]
        if "Data-Analysis-New-Grad" in url:
            return """
| Company | Job Title | Location | Work Model | Date Posted |
| ----- | --------- |  --------- | ---- | ------- |
| **[Company A](https://companya.com)** | **[Data Analyst (Entry Level)](https://example.com/a)** | NYC | Hybrid | Mar 11 |
"""
        return None

    monkeypatch.setattr(jp, "_fetch_markdown", fake_fetch_markdown, raising=False)

    # First run: should see 1 new job from table
    new_count, total = jp.run_job_scan(model_name="dummy-model")
    assert new_count == 1
    assert total == 1

    # Second run: no new jobs should be added (seen_ids state used)
    new_count2, total2 = jp.run_job_scan(model_name="dummy-model")
    assert new_count2 == 0
    assert total2 == 1

    # Results file should contain job from table (job posting URL used)
    data = json.loads((data_dir / "job_results.json").read_text(encoding="utf-8"))
    jobs = data["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["url"] == "https://example.com/a"
    assert "Company A" in jobs[0]["title"]

