"""Create synthetic demo PDFs; NEVER represent these as employer-provided policies.

Run: python scripts/create_demo_docs.py [--output docs]
"""
import argparse
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

SAMPLE_TEXT: dict[str, str] = {
    "hr/leave_policy.pdf": (
        "SYNTHETIC TEST FIXTURE - NOT A REAL COMPANY POLICY.\n"
        "Leave policy. All employees can request vacation leave through the sample employee portal. "
        "Demo leave allowance is twenty-two working days per year, subject to the illustrative sample approval workflow. "
        "These figures are invented solely to test retrieval and must not be relied upon."
    ),
    "hr/code_of_conduct.pdf": (
        "SYNTHETIC TEST FIXTURE - NOT A REAL COMPANY POLICY.\n"
        "Code of conduct. Employees must report sample confidentiality incidents to the example compliance desk. "
        "This fictional policy also prohibits retaliation against good faith whistleblowers."
    ),
    "hr/performance_review.pdf": (
        "SYNTHETIC TEST FIXTURE - MANAGER RESTRICTED - NOT A REAL COMPANY POLICY.\n"
        "Performance review. Managers conduct a fictional quarterly calibration meeting. "
        "Performance rating scale in this demo is one through five. The sample promotion committee meets in October."
    ),
    "finance/expense_policy.pdf": (
        "SYNTHETIC TEST FIXTURE - NOT A REAL COMPANY POLICY.\n"
        "Expense policy. Example finance claims require receipts and an imaginary accounting code. "
        "This sample document uses fictional reimbursement limits."
    ),
    "finance/travel_reimbursement.pdf": (
        "SYNTHETIC TEST FIXTURE - NOT A REAL COMPANY POLICY.\n"
        "Travel reimbursement. For a fictitious trip, submit boarding passes with a mocked claim form. "
        "Sample finance reviewers approve travel claims after verifying receipts."
    ),
    "exec/compensation_committee.pdf": (
        "SYNTHETIC TEST FIXTURE - EXECUTIVE RESTRICTED - NOT A REAL COMPANY POLICY.\n"
        "Compensation committee. A fabricated executive bonus strategy is reviewed confidentially each quarter. "
        "This information is purely illustrative for RBAC tests."
    ),
    "exec/strategic_plan.pdf": (
        "SYNTHETIC TEST FIXTURE - EXECUTIVE RESTRICTED - SCANNED - NOT A REAL COMPANY POLICY.\n"
        "Strategic plan. A fictional board intends to expand the imaginary Redwood product in 2028. "
        "This page is intentionally image-only so that OCR is required."
    ),
}


def create_demo_pdfs(root: Path) -> None:
    for filename, text in SAMPLE_TEXT.items():
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing PDF: {path}")
        if filename == "exec/strategic_plan.pdf":
            image = Image.new("RGB", (1500, 1800), "white")
            draw = ImageDraw.Draw(image)
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            try:
                font = ImageFont.truetype(font_path, 31)
            except OSError:
                font = ImageFont.load_default()
            # Fixed lines with ample spacing result in reliably OCR-readable synthetic scan.
            import textwrap
            y = 80
            for line in textwrap.wrap(text, width=63):
                draw.text((65, y), line, fill="black", font=font)
                y += 65
            image.save(path, "PDF", resolution=160)
        else:
            pdf = canvas.Canvas(str(path), pagesize=letter)
            pdf.setFont("Helvetica", 11)
            t = pdf.beginText(45, 750)
            import textwrap
            for line in textwrap.wrap(text, width=90):
                t.textLine(line)
            pdf.drawText(t)
            pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs"))
    args = parser.parse_args()
    create_demo_pdfs(args.output)
    print(f"Created {len(SAMPLE_TEXT)} explicitly fictional demo PDFs in {args.output}")


if __name__ == "__main__":
    main()
