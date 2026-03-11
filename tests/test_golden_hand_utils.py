from io import BytesIO

import pytest

# Skip this whole test module if PyMuPDF/fitz (and therefore app.py) are not importable
pytest.importorskip("fitz")
import app


def test_ensure_noindent_before_roles_inserts_marker():
    latex = (
        "\\section*{Professional Experience}\n"
        "\\rule{\\textwidth}{0.4pt}\n"
        "\\small\n"
        "\\textbf{Company A, City} \\hfill 2024--Present\\\\\n"
        "\\textbf{\\underline{Role A}}\n"
        "\\begin{itemize}\n"
        "  \\item Bullet one\n"
        "\\end{itemize}\n"
    )
    out = app._ensure_noindent_before_roles(latex)
    # There should be a \\noindent line immediately before the \\textbf{Company A...} line
    lines = out.split("\n")
    idx = [i for i, l in enumerate(lines) if "Company A, City" in l][0]
    assert lines[idx - 1].strip() == "\\noindent"


def test_clean_latex_experience_escapes_percent_and_strips_comments():
    latex = (
        "\\begin{verbatim}\n"
        "verbatim stuff\n"
        "\\end{verbatim}\n"
        "% added JD-focused bullet\n"
        "\\item Improved accuracy by 50% (X)\n"
    )
    cleaned = app._clean_latex_experience(latex)
    assert "verbatim" not in cleaned
    assert "% added JD-focused" not in cleaned
    # 50% should become 50\\% to survive LaTeX
    assert "50\\%" in cleaned
    # (X) label removed
    assert "(X)" not in cleaned


def test_ensure_data_science_intern_three_bullets_adds_third_when_missing(monkeypatch):
    latex = (
        "\\textbf{Apptware Pvt. Ltd., Pune} \\\\ \n"
        "\\textbf{\\underline{Data Science Intern}}\n"
        "\\begin{itemize}[left=0pt, itemsep=0pt]\n"
        " \\item First bullet\n"
        " \\item Second bullet\n"
        "\\end{itemize}\n"
    )

    class FakeRespBlock:
        def __init__(self, text: str):
            self.type = "output_text"
            self.text = text

    class FakeRespMsg:
        def __init__(self, text: str):
            self.type = "message"
            self.content = [FakeRespBlock(text)]

    class FakeResponse:
        def __init__(self, text: str):
            self.output = [FakeRespMsg(text)]

    class FakeClient:
        class Responses:
            @staticmethod
            def create(model: str, input):
                return FakeResponse("Accomplished X as measured by Y, by doing Z.")

        responses = Responses()

    # Patch client used inside app module
    monkeypatch.setattr(app, "client", FakeClient(), raising=False)

    out = app._ensure_data_science_intern_three_bullets(
        latex,
        portfolio_text="",
        job_description="Data Science Intern working with Python.",
        model_name="dummy-model",
    )
    # There should now be 3 \\item lines in the Data Science Intern block
    block = out[out.find("\\begin{itemize}"): out.find("\\end{itemize}")]
    assert block.count("\\item") == 3

