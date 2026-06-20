import glob
import re
import fitz  # pymupdf — better text extraction than pypdf, especially for math-adjacent text


_CAPTION = re.compile(r"^(Figure|Table|Algorithm)\s+\d+[:\.].*", re.IGNORECASE | re.MULTILINE)
_CITATIONS = re.compile(r"\[\d+(?:[,;\s]+\d+)*\]")  # [13], [35, 2, 5], [1; 2]
_LONELY_NUMBERS = re.compile(r"(?<!\w)(\d{1,3})\n")  # standalone page/footnote numbers


def _clean(text):
    text = _CAPTION.sub("", text)
    text = _CITATIONS.sub("", text)
    text = _LONELY_NUMBERS.sub("", text)
    return text.strip()


def _extract_text(path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text and text.strip():
            text = _clean(text)
            pages.append({"page_content": text, "metadata": {"source": path, "page": i}})
    doc.close()
    return pages


def load_pdf(path):
    return _extract_text(path)


def load_all_pdfs(folder):
    """Returns one entry per PDF with all pages joined.

    Joining pages lets chunkers see across page boundaries — the most common
    cause of split explanations and low retrieval scores.
    """
    all_docs = []
    for path in sorted(glob.glob(f"{folder}/*.pdf")):
        pages = _extract_text(path)
        if not pages:
            continue
        full_text = "\n".join(p["page_content"] for p in pages)
        all_docs.append({
            "page_content": full_text,
            "metadata": {"source": path, "page": 0},
        })
    return all_docs
