import json
from pathlib import Path

import pytest

import jobs_pipeline as jp


def test_is_entry_level_line_filters_internships():
    assert jp._is_entry_level_line("Great New Grad Data Scientist (Entry Level)") is True
    assert jp._is_entry_level_line("Summer 2026 Software Engineering Intern") is False
    assert jp._is_entry_level_line("Data Scientist (Co-op)") is False


def test_github_to_raw_readme_basic():
    url = "https://github.com/jobright-ai/2026-Data-Analysis-New-Grad"
    raw = jp._github_to_raw_readme(url)
    assert raw == (
        "https://raw.githubusercontent.com/jobright-ai/2026-Data-Analysis-New-Grad/HEAD/README.md"
    )


def test_parse_jobs_only_keeps_entry_level():
    md = """
    - [Awesome Company - Data Scientist (Entry Level)](https://example.com/job1)
    - [Another Co - SWE Intern](https://example.com/job2)
    - [Third Co - New Grad ML Engineer](https://example.com/job3)
    """
    jobs = jp._parse_jobs(md, source_repo="demo-repo")
    titles = [j.title for j in jobs]
    assert "Awesome Company - Data Scientist (Entry Level)" in titles
    assert "Third Co - New Grad ML Engineer" in titles
    # Internship should be filtered out
    assert all("Intern" not in t for t in titles)


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

    # Fake markdown content for the New-Grad repo
    def fake_fetch_markdown(url: str, timeout: int = 10) -> str | None:  # type: ignore[override]
        if "Data-Analysis-New-Grad" in url:
            return """
            - [Company A - Data Analyst (Entry Level)](https://example.com/a)
            - [Company B - Data Analyst Intern](https://example.com/b)
            """
        return None

    monkeypatch.setattr(jp, "_fetch_markdown", fake_fetch_markdown, raising=False)

    # First run: should see 1 new job (internship filtered)
    new_count, total = jp.run_job_scan(model_name="dummy-model")
    assert new_count == 1
    assert total == 1

    # Second run: no new jobs should be added (seen_ids state used)
    new_count2, total2 = jp.run_job_scan(model_name="dummy-model")
    assert new_count2 == 0
    assert total2 == 1

    # Results file should contain our filtered job (no internships)
    data = json.loads((data_dir / "job_results.json").read_text(encoding="utf-8"))
    jobs = data["jobs"]
    assert len(jobs) == 1
    assert "Data Analyst Intern" not in jobs[0]["title"]

