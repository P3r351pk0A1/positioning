"""Export project Markdown documentation to PDF.

The script intentionally keeps dependencies small: it uses ReportLab and
system fonts with Cyrillic support. It is not a full Markdown renderer; it
supports the subset used in the project documentation.
"""

from __future__ import annotations

import argparse
import os
import re
import textwrap
from html import escape
from pathlib import Path

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "PROJECT_OVERVIEW.md",
    PROJECT_ROOT / "docs" / "USER_GUIDE.md",
    PROJECT_ROOT / "docs" / "ALGORITHM_TDOA_DETAILED.md",
    PROJECT_ROOT / "docs" / "MODEL_STATUS.md",
    PROJECT_ROOT / "docs" / "PMI_PROTOCOL_SUMMARY.md",
]


def register_fonts() -> tuple[str, str, str]:
    fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    regular = fonts_dir / "arial.ttf"
    bold = fonts_dir / "arialbd.ttf"
    mono = fonts_dir / "consola.ttf"

    if not regular.exists():
        raise FileNotFoundError(f"Не найден шрифт для кириллицы: {regular}")

    pdfmetrics.registerFont(TTFont("DocRegular", regular))
    if bold.exists():
        pdfmetrics.registerFont(TTFont("DocBold", bold))
    else:
        pdfmetrics.registerFont(TTFont("DocBold", regular))
    if mono.exists():
        pdfmetrics.registerFont(TTFont("DocMono", mono))
    else:
        pdfmetrics.registerFont(TTFont("DocMono", regular))
    registerFontFamily("DocRegular", normal="DocRegular", bold="DocBold")
    return "DocRegular", "DocBold", "DocMono"


def build_styles() -> dict[str, ParagraphStyle]:
    regular, bold, mono = register_fonts()
    base = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=bold,
            fontSize=20,
            leading=24,
            spaceAfter=8,
            spaceBefore=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=bold,
            fontSize=15,
            leading=18,
            spaceAfter=6,
            spaceBefore=10,
        ),
        "h3": ParagraphStyle(
            "h3",
            parent=base["Heading3"],
            fontName=bold,
            fontSize=12,
            leading=15,
            spaceAfter=5,
            spaceBefore=8,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=10,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=10,
            leading=14,
            leftIndent=8 * mm,
            firstLineIndent=-4 * mm,
            spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "code",
            parent=base["Code"],
            fontName=mono,
            fontSize=8,
            leading=10,
            leftIndent=4 * mm,
            rightIndent=4 * mm,
            spaceBefore=3,
            spaceAfter=6,
        ),
    }


def convert_inline_markdown(text: str) -> str:
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r'<font name="DocMono">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def split_paragraphs(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line.strip():
            if current:
                paragraphs.append(" ".join(item.strip() for item in current))
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(item.strip() for item in current))
    return paragraphs


def markdown_to_flowables(markdown: str, styles: dict[str, ParagraphStyle]) -> list:
    flowables: list = []
    paragraph_buffer: list[str] = []
    code_buffer: list[str] = []
    in_code = False

    def flush_paragraphs() -> None:
        nonlocal paragraph_buffer
        for paragraph in split_paragraphs(paragraph_buffer):
            flowables.append(Paragraph(convert_inline_markdown(paragraph), styles["body"]))
        paragraph_buffer = []

    def flush_code() -> None:
        nonlocal code_buffer
        if not code_buffer:
            return
        wrapped: list[str] = []
        for line in code_buffer:
            if len(line) <= 92:
                wrapped.append(line)
            else:
                wrapped.extend(textwrap.wrap(line, width=92, replace_whitespace=False))
        flowables.append(Preformatted("\n".join(wrapped), styles["code"], maxLineLength=92))
        code_buffer = []

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_paragraphs()
                in_code = True
            continue

        if in_code:
            code_buffer.append(line)
            continue

        if line.startswith("# "):
            flush_paragraphs()
            flowables.append(Paragraph(convert_inline_markdown(line[2:].strip()), styles["h1"]))
            continue
        if line.startswith("## "):
            flush_paragraphs()
            flowables.append(Paragraph(convert_inline_markdown(line[3:].strip()), styles["h2"]))
            continue
        if line.startswith("### "):
            flush_paragraphs()
            flowables.append(Paragraph(convert_inline_markdown(line[4:].strip()), styles["h3"]))
            continue
        if line.startswith("- "):
            flush_paragraphs()
            flowables.append(Paragraph("- " + convert_inline_markdown(line[2:].strip()), styles["bullet"]))
            continue
        if re.match(r"^\d+\. ", line):
            flush_paragraphs()
            flowables.append(Paragraph(convert_inline_markdown(line.strip()), styles["bullet"]))
            continue
        if line.startswith("|"):
            flush_paragraphs()
            if set(line.replace("|", "").replace(" ", "")) <= {"-", ":"}:
                continue
            flowables.append(Preformatted(line, styles["code"], maxLineLength=92))
            continue

        paragraph_buffer.append(line)

    flush_paragraphs()
    flush_code()
    return flowables


def export_pdf(source: Path, output: Path, styles: dict[str, ParagraphStyle]) -> None:
    markdown = source.read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=source.stem,
        author="positioning",
    )
    story = markdown_to_flowables(markdown, styles)
    doc.build(story)


def export_combined_pdf(sources: list[Path], output: Path, styles: dict[str, ParagraphStyle]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Positioning documentation",
        author="positioning",
    )
    story: list = []
    for index, source in enumerate(sources):
        if index:
            story.append(PageBreak())
        story.extend(markdown_to_flowables(source.read_text(encoding="utf-8"), styles))
        story.append(Spacer(1, 4 * mm))
    doc.build(story)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export project Markdown documentation to PDF.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "docs" / "pdf"))
    parser.add_argument("--combined-name", default="positioning_documentation.pdf")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    styles = build_styles()
    sources = [source for source in DEFAULT_SOURCES if source.exists()]
    for source in sources:
        output = output_dir / f"{source.stem}.pdf"
        export_pdf(source, output, styles)
        print(f"exported {output}")

    combined = output_dir / args.combined_name
    export_combined_pdf(sources, combined, styles)
    print(f"exported {combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
