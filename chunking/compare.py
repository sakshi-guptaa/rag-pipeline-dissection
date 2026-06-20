"""
Chunking Comparison

Runs all 4 chunkers on the same PDF and prints a comparison table
with chunk counts, size stats, and timing.

Usage:
    python chunking/compare.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.loader import load_all_pdfs
from chunking import recursive, character, section_wise, semantic

PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")


def stats(chunks):
    sizes = [len(c["text"]) for c in chunks]
    if not sizes:
        return {"count": 0, "avg": 0, "min": 0, "max": 0}
    return {
        "count": len(sizes),
        "avg": sum(sizes) / len(sizes),
        "min": min(sizes),
        "max": max(sizes),
    }


def run_chunker(name, fn, pages, **kwargs):
    start = time.perf_counter()
    chunks = fn(pages, **kwargs)
    elapsed = time.perf_counter() - start
    return chunks, elapsed


def main():
    pages = load_all_pdfs(PAPERS_DIR)
    print(f"Loaded {len(pages)} pages from {PAPERS_DIR}\n")

    chunkers = [
        ("Recursive (800c)", recursive.chunk, {"chunk_size": 800, "chunk_overlap": 80}),
        ("Character (800c)", character.chunk, {"chunk_size": 800, "chunk_overlap": 80}),
        ("Section-wise (800c)", section_wise.chunk, {"chunk_size": 800, "chunk_overlap": 80}),
        ("Semantic", semantic.chunk, {"chunk_size": 800, "chunk_overlap": 80}),
    ]

    print(f"{'Chunker':<22} {'Chunks':>7} {'Avg':>7} {'Min':>6} {'Max':>6} {'Time':>8}")
    print("-" * 62)

    all_results = {}
    for name, fn, kwargs in chunkers:
        chunks, elapsed = run_chunker(name, fn, pages, **kwargs)
        s = stats(chunks)
        all_results[name] = chunks
        print(
            f"{name:<22} {s['count']:>7} {s['avg']:>7.0f} {s['min']:>6} {s['max']:>6} {elapsed:>7.3f}s"
        )

    print("\n--- Sample chunks (first from each) ---\n")
    for name, chunks in all_results.items():
        if chunks:
            preview = chunks[0]["text"][:200].replace("\n", " ")
            meta = {k: v for k, v in chunks[0]["metadata"].items() if k != "chunker"}
            print(f"[{name}] ({len(chunks[0]['text'])} chars)")
            print(f"  {preview}...")
            print(f"  metadata: {meta}\n")


if __name__ == "__main__":
    main()
