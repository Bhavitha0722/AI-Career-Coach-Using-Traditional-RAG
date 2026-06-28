from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
import docx2txt

# This file is responsible for the different type of file types 
def read_uploaded_file(uploaded_file) -> str:
    """Read txt, pdf, or docx uploaded from Streamlit."""
    if uploaded_file is None:
        return ""

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:  #temporary file
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    if suffix == ".pdf":
        reader = PdfReader(tmp_path)
        text = []
        for page in reader.pages:  # reads page by page 
            text.append(page.extract_text() or "")
        return "\n".join(text) # reads page by page and convert into single text

    if suffix == ".docx":
        return docx2txt.process(tmp_path)

    raise ValueError("Unsupported file type. Please upload .txt, .pdf, or .docx")
