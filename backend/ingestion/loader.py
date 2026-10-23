import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class TextSection:
    text: str
    page_number: int | None = None


def safe_filename(filename: str) -> str:
    # Both separator styles must be handled, even when running on macOS.
    name = filename.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^\w. ()-]", "_", name)[:180]
    if not name or name in {".", ".."}:
        raise ValueError("Invalid filename")
    if Path(name).suffix.lower() not in SUPPORTED:
        raise ValueError("Supported formats: PDF, DOCX, TXT, MD")
    return name


def extract(path: Path, suffix: str) -> list[TextSection]:
    try:
        if suffix == ".pdf":
            import pymupdf

            with pymupdf.open(path, filetype="pdf") as document:
                if document.needs_pass:
                    raise ValueError("Password-protected PDFs are unsupported")
                sections = [TextSection(page.get_text(), i + 1) for i, page in enumerate(document)]
        elif suffix == ".docx":
            from docx import Document

            document = Document(path)
            # iter_inner_content preserves paragraph/table order. DOCX has no reliable page numbers.
            parts = []
            from docx.table import Table

            for block in document.iter_inner_content():
                if isinstance(block, Table):
                    parts.extend(" | ".join(cell.text for cell in row.cells) for row in block.rows)
                else:
                    parts.append(block.text)
            sections = [TextSection("\n".join(parts))]
        else:
            sections = [TextSection(path.read_text(encoding="utf-8-sig"))]
    except ValueError:
        raise
    except Exception:
        raise ValueError("Cannot parse document; check format, encoding, and corruption") from None
    sections = [s for s in sections if s.text.strip()]
    if not sections:
        raise ValueError("No extractable text. Scanned PDFs require OCR, which is not included")
    return sections
