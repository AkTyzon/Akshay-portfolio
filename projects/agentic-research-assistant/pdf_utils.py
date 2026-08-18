import fitz  # PyMuPDF
import re

def extract_text_from_pdf(uploaded_files) -> str:
    """Extract text from multiple PDFs."""
    text = []
    for f in uploaded_files:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        for page in doc:
            text.append(page.get_text())
    return re.sub(r"\s+", " ", " ".join(text)).strip()
