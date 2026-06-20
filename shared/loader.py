import glob
from pypdf import PdfReader


def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page_content": text, "metadata": {"source": path, "page": i}})
    return pages


def load_all_pdfs(folder):
    docs = []
    for path in sorted(glob.glob(f"{folder}/*.pdf")):
        docs.extend(load_pdf(path))
    return docs
