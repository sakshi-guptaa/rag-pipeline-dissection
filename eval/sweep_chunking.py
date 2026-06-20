"""
eval/sweep_chunking.py — Phase 2 chunking sweep.

Runs all 4 chunkers × 6 chunk sizes in one pass.
Pages and embedding model are loaded once and reused across all combinations.

Usage:
    python eval/sweep_chunking.py [--save]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.loader import load_all_pdfs
from shared.embedder import embed_texts, embed_query, get_model
from vectordb.faiss_store import FaissStore
from chunking import recursive, character, section_wise, semantic

PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")
GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json")
FINDINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FINDINGS.md")

CHUNKERS = {
    "recursive": recursive.chunk,
    "character": character.chunk,
    "section_wise": section_wise.chunk,
    "semantic": semantic.chunk,
}

CHUNK_SIZES = [200, 400, 600, 800, 1000, 1200]


def recall_at_k(hits, evidence, k):
    return 1.0 if any(evidence.lower() in h["text"].lower() for h in hits[:k]) else 0.0


def reciprocal_rank(hits, evidence, k):
    for i, h in enumerate(hits[:k]):
        if evidence.lower() in h["text"].lower():
            return 1.0 / (i + 1)
    return 0.0


def evaluate_combo(chunker_name, chunker_fn, chunk_size, pages, golden, tokenizer):
    overlap = max(20, chunk_size // 10)
    chunk_kwargs = {"chunk_size": chunk_size, "chunk_overlap": overlap}

    t0 = time.perf_counter()
    chunks = chunker_fn(pages, **chunk_kwargs)
    chunk_time = (time.perf_counter() - t0) * 1000

    if not chunks:
        return None

    texts = [c["text"] for c in chunks]

    t0 = time.perf_counter()
    embeddings = embed_texts(texts)
    embed_time = (time.perf_counter() - t0) * 1000

    dim = embeddings.shape[1]
    store = FaissStore(dimension=dim)
    store.add(chunks, embeddings)

    r1s, r3s, r5s, mrrs, latencies = [], [], [], [], []
    for item in golden:
        qe = embed_query(item["question"])
        t0 = time.perf_counter()
        hits = store.search(qe, k=5)
        latencies.append((time.perf_counter() - t0) * 1000)
        r1s.append(recall_at_k(hits, item["evidence"], 1))
        r3s.append(recall_at_k(hits, item["evidence"], 3))
        r5s.append(recall_at_k(hits, item["evidence"], 5))
        mrrs.append(reciprocal_rank(hits, item["evidence"], 5))

    sizes_chars = [len(c["text"]) for c in chunks]
    token_counts = [len(tokenizer.tokenize(c["text"])) for c in chunks]
    oversized = sum(1 for t in token_counts if t > 384)

    return {
        "chunker": chunker_name,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "recall_at_1": round(np.mean(r1s), 4),
        "recall_at_3": round(np.mean(r3s), 4),
        "recall_at_5": round(np.mean(r5s), 4),
        "mrr": round(np.mean(mrrs), 4),
        "count": len(chunks),
        "avg_chars": int(np.mean(sizes_chars)),
        "oversized_pct": round(oversized / len(chunks) * 100, 1),
        "token_util_pct": round(np.mean([min(t / 384, 1.0) for t in token_counts]) * 100, 1),
        "embed_time_ms": round(embed_time, 0),
        "avg_latency_ms": round(np.mean(latencies), 2),
    }


def print_table(results):
    header = f"\n{'Chunker':<14} {'Size':>5} {'Ovlp':>5} | {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>7} | {'Chunks':>7} {'AvgChr':>7} {'Oversz':>7} {'TokUt':>6} | {'EmbMs':>7}"
    print(header)
    print("─" * len(header))

    best_r5 = max(r["recall_at_5"] for r in results)
    for r in results:
        marker = " ◀ BEST" if r["recall_at_5"] == best_r5 else ""
        print(
            f"{r['chunker']:<14} {r['chunk_size']:>5} {r['overlap']:>5} | "
            f"{r['recall_at_1']:>5.0%} {r['recall_at_3']:>5.0%} {r['recall_at_5']:>5.0%} {r['mrr']:>7.4f} | "
            f"{r['count']:>7} {r['avg_chars']:>7} {r['oversized_pct']:>6.1f}% {r['token_util_pct']:>5.1f}% | "
            f"{r['embed_time_ms']:>6.0f}ms{marker}"
        )


def save_to_findings(results):
    best = max(results, key=lambda r: (r["recall_at_5"], r["mrr"]))
    rows = ""
    for r in results:
        rows += (
            f"| {r['chunker']} | size={r['chunk_size']} overlap={r['overlap']} | faiss "
            f"| {r['recall_at_1']:.0%} | {r['recall_at_3']:.0%} | {r['recall_at_5']:.0%} "
            f"| {r['mrr']:.4f} | — | {r['count']} | {r['avg_chars']} | {r['oversized_pct']}% "
            f"| {r['token_util_pct']}% | {r['embed_time_ms']:.0f}ms | {r['avg_latency_ms']:.2f}ms |\n"
        )

    with open(FINDINGS_PATH, "a") as f:
        f.write("\n\n## Phase 2 — Chunking Sweep Raw Results\n\n")
        f.write("| Chunker | Config | Store | R@1 | R@3 | R@5 | MRR | Top-1 | Chunks | Avg Size | Oversized | Token Util | Embed Time | Query Latency |\n")
        f.write("|---------|--------|-------|-----|-----|-----|-----|-------|--------|----------|-----------|------------|------------|---------------|\n")
        f.write(rows)
        f.write(f"\n**Winner:** `{best['chunker']}` · size={best['chunk_size']} · overlap={best['overlap']} → Recall@5={best['recall_at_5']:.0%}, MRR={best['mrr']:.4f}\n")

    print(f"\n  → Saved to FINDINGS.md")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    print("Loading pages and model (once)...")
    pages = load_all_pdfs(PAPERS_DIR)
    tokenizer = get_model().tokenizer
    print(f"  {len(pages)} documents loaded\n")

    results = []
    total = len(CHUNKERS) * len(CHUNK_SIZES)
    done = 0

    for chunker_name, chunker_fn in CHUNKERS.items():
        for chunk_size in CHUNK_SIZES:
            done += 1
            print(f"[{done:>2}/{total}] {chunker_name} size={chunk_size}...", end=" ", flush=True)
            try:
                r = evaluate_combo(chunker_name, chunker_fn, chunk_size, pages, golden, tokenizer)
                if r:
                    results.append(r)
                    print(f"R@5={r['recall_at_5']:.0%}  MRR={r['mrr']:.4f}")
                else:
                    print("skipped (0 chunks)")
            except Exception as e:
                print(f"ERROR: {e}")

    print_table(results)

    if args.save:
        best = save_to_findings(results)
        print(f"\nBest combo: {best['chunker']} size={best['chunk_size']} → Recall@5={best['recall_at_5']:.0%}, MRR={best['mrr']:.4f}")


if __name__ == "__main__":
    main()
