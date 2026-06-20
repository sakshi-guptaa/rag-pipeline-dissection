"""
Character Text Splitter

Splits only on a single separator (default: paragraph break).
Merges small pieces up toward chunk_size, but ships oversized
chunks as-is when a paragraph exceeds the limit.
"""


def chunk(pages, chunk_size=800, chunk_overlap=80, separator="\n\n"):
    results = []
    for page in pages:
        text = page["page_content"]
        pieces = text.split(separator)

        current = ""
        prev_tail = ""
        for piece in pieces:
            candidate = piece if not current else current + separator + piece
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    results.append({
                        "text": current.strip(),
                        "metadata": {**page["metadata"], "chunker": "character"},
                    })
                    prev_tail = current[-chunk_overlap:] if chunk_overlap > 0 else ""
                if len(piece) > chunk_size:
                    results.append({
                        "text": (prev_tail + piece).strip(),
                        "metadata": {
                            **page["metadata"],
                            "chunker": "character",
                            "oversized": True,
                        },
                    })
                    prev_tail = piece[-chunk_overlap:] if chunk_overlap > 0 else ""
                    current = ""
                else:
                    current = prev_tail + piece
                    prev_tail = ""
        if current.strip():
            results.append({
                "text": current.strip(),
                "metadata": {**page["metadata"], "chunker": "character"},
            })
    return [r for r in results if len(r["text"]) >= 20]
