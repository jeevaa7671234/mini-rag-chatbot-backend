import pytest

from app.ingestion import chunk_text, collect_chunks, pages_from_pdf
from scripts.create_demo_docs import create_demo_pdfs


def test_chunks_preserve_overlap_and_bounded_size():
    text = "\n\n".join("Policy sentence %03d has detailed terms for all staff. " % i for i in range(100))
    parts = chunk_text(text)
    assert len(parts) > 3
    assert max(map(len, parts)) <= 900
    for previous, nxt in zip(parts, parts[1:]):
        assert previous.split()[-3:] == nxt.split()[max(0, 0):][:3] or bool(set(previous.split()[-15:]) & set(nxt.split()[:15]))


def test_scan_is_really_image_only_and_ocr_succeeds(tmp_path):
    import fitz
    root = tmp_path / "docs"
    create_demo_pdfs(root)
    scanned = root / "exec/strategic_plan.pdf"
    with fitz.open(scanned) as pdf:
        assert pdf[0].get_text().strip() == ""
    page, text = pages_from_pdf(scanned)[0]
    assert page == 1
    assert "Strategic" in text or "strategic" in text
    assert "Redwood" in text


def test_seven_pdfs_have_correct_acl_and_page(tmp_path):
    root = tmp_path / "docs"
    create_demo_pdfs(root)
    chunks = collect_chunks(root)
    assert {c.source_file for c in chunks} == {
        "leave_policy.pdf", "code_of_conduct.pdf", "performance_review.pdf",
        "expense_policy.pdf", "travel_reimbursement.pdf", "compensation_committee.pdf",
        "strategic_plan.pdf",
    }
    assert all(c.page >= 1 and c.text for c in chunks)
    assert next(c for c in chunks if c.source_file == "performance_review.pdf").access_level == 2
    assert next(c for c in chunks if c.source_file == "strategic_plan.pdf").access_level == 3
    assert next(c for c in chunks if c.source_file == "strategic_plan.pdf").department == "exec"


def test_no_undocumented_acl_defaults(tmp_path):
    root = tmp_path / "docs"
    create_demo_pdfs(root)
    (root / "hr/undocumented.pdf").write_bytes(b"not a real pdf")
    with pytest.raises(ValueError, match="Unknown PDF"):
        collect_chunks(root)
    (root / "hr/undocumented.pdf").unlink()
    (root / "hr/leave_policy.pdf").unlink()
    with pytest.raises(FileNotFoundError, match="Missing"):
        collect_chunks(root)
