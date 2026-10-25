from pathlib import Path
from pypdf import PdfReader


def load_pdf(path: Path) -> list[str]:
    """Return raw text per page."""
    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]