"""
eval/run_phase.py — Standardized RAG pipeline evaluator.

Every phase runs this same script. Only the modules under test change.
Results are printed to stdout and optionally appended to FINDINGS.md.

Usage:
    python eval/run_phase.py \\
        --phase "0-Baseline" \\
        --chunker recursive \\
        --chunk-size 800 \\
        --chunk-overlap 80 \\
        --store faiss \\
        --top-k 5 \\
        --note "Default settings, no optimisation"
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.loader import load_all_pdfs
from shared.embedder import embed_texts, embed_query, get_model
from vectordb.faiss_store import FaissStore
from vectordb.qdrant_store import QdrantStore
from vectordb.chroma_store import ChromaStore
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

STORES = {
    "faiss": lambda dim: FaissStore(dimension=dim),
    "qdrant": lambda dim: QdrantStore(collection_name="eval", dimension=dim),
    "chroma": lambda dim: ChromaStore(collection_name="eval"),
}


def token_count(text, tokenizer):
    return len(tokenizer.tokenize(text))


def chunk_stats(chunks, tokenizer, chunk_size):
    sizes_chars = [len(c["text"]) for c in chunks]
    token_counts = [token_count(c["text"], tokenizer) for c in chunks]
    model_limit = 384
    oversized = sum(1 for t in token_counts if t > model_limit)
    avg_utilization = np.mean([min(t / model_limit, 1.0) for t in token_counts]) * 100

    return {
        "count": len(chunks),
        "avg_chars": int(np.mean(sizes_chars)),
        "std_chars": int(np.std(sizes_chars)),
        "min_chars": min(sizes_chars),
        "max_chars": max(sizes_chars),
        "avg_tokens": int(np.mean(token_counts)),
        "oversized_count": oversized,
        "oversized_pct": round(oversized / len(chunks) * 100, 1),
        "token_utilization_pct": round(avg_utilization, 1),
    }


def recall_at_k(hits, evidence, k):
    for h in hits[:k]:
        if evidence.lower() in h["text"].lower():
            return 1.0
    return 0.0


def reciprocal_rank(hits, evidence, k):
    for i, h in enumerate(hits[:k]):
        if evidence.lower() in h["text"].lower():
            return 1.0 / (i + 1)
    return 0.0


def run_evaluation(chunker_fn, chunk_kwargs, store_fn, golden, top_k=5):
    # Parse
    t0 = time.perf_counter()
    pages = load_all_pdfs(PAPERS_DIR)
    parse_time = (time.perf_counter() - t0) * 1000

    # Chunk
    t0 = time.perf_counter()
    chunks = chunker_fn(pages, **chunk_kwargs)
    chunk_time = (time.perf_counter() - t0) * 1000

    if not chunks:
        raise ValueError("Chunker produced 0 chunks")

    texts = [c["text"] for c in chunks]

    # Embed
    t0 = time.perf_counter()
    embeddings = embed_texts(texts)
    embed_time = (time.perf_counter() - t0) * 1000

    dim = embeddings.shape[1]

    # Index
    store = store_fn(dim)
    t0 = time.perf_counter()
    store.add(chunks, embeddings)
    index_time = (time.perf_counter() - t0) * 1000

    # Evaluate over golden set
    r1s, r3s, r5s, mrrs, top1_scores, latencies = [], [], [], [], [], []

    for item in golden:
        qe = embed_query(item["question"])

        t0 = time.perf_counter()
        hits = store.search(qe, k=max(top_k, 5))
        latencies.append((time.perf_counter() - t0) * 1000)

        r1s.append(recall_at_k(hits, item["evidence"], 1))
        r3s.append(recall_at_k(hits, item["evidence"], 3))
        r5s.append(recall_at_k(hits, item["evidence"], 5))
        mrrs.append(reciprocal_rank(hits, item["evidence"], 5))

        if hits:
            top1_scores.append(hits[0]["score"])

    # Chunk stats
    tokenizer = get_model().tokenizer
    stats = chunk_stats(chunks, tokenizer, chunk_kwargs.get("chunk_size", 800))

    return {
        "recall_at_1": round(np.mean(r1s), 4),
        "recall_at_3": round(np.mean(r3s), 4),
        "recall_at_5": round(np.mean(r5s), 4),
        "mrr": round(np.mean(mrrs), 4),
        "avg_top1_score": round(np.mean(top1_scores), 4),
        "parse_time_ms": round(parse_time, 1),
        "chunk_time_ms": round(chunk_time, 1),
        "embed_time_ms": round(embed_time, 1),
        "index_time_ms": round(index_time, 1),
        "avg_query_latency_ms": round(np.mean(latencies), 2),
        **stats,
    }


def print_results(phase, note, chunker, chunk_size, chunk_overlap, store, results):
    print(f"\n{'='*60}")
    print(f"  Phase {phase}")
    print(f"  {note}")
    print(f"{'='*60}")
    print(f"  Config : {chunker} chunker | size={chunk_size} | overlap={chunk_overlap} | {store} store")
    print(f"{'─'*60}")
    print(f"  RETRIEVAL QUALITY")
    print(f"    Recall@1          : {results['recall_at_1']:.2%}")
    print(f"    Recall@3          : {results['recall_at_3']:.2%}")
    print(f"    Recall@5          : {results['recall_at_5']:.2%}")
    print(f"    MRR               : {results['mrr']:.4f}")
    print(f"    Avg Top-1 Score   : {results['avg_top1_score']:.4f}")
    print(f"{'─'*60}")
    print(f"  CHUNK QUALITY")
    print(f"    Total chunks      : {results['count']}")
    print(f"    Avg size          : {results['avg_chars']} chars / {results['avg_tokens']} tokens")
    print(f"    Std dev           : {results['std_chars']} chars")
    print(f"    Min / Max         : {results['min_chars']} / {results['max_chars']} chars")
    print(f"    Oversized (>384t) : {results['oversized_count']} ({results['oversized_pct']}%)")
    print(f"    Token utilization : {results['token_utilization_pct']}%")
    print(f"{'─'*60}")
    print(f"  TIMING")
    print(f"    Parse             : {results['parse_time_ms']:.1f} ms")
    print(f"    Chunk             : {results['chunk_time_ms']:.1f} ms")
    print(f"    Embed             : {results['embed_time_ms']:.1f} ms")
    print(f"    Index             : {results['index_time_ms']:.1f} ms")
    print(f"    Avg query latency : {results['avg_query_latency_ms']:.2f} ms")
    print(f"{'='*60}\n")


def append_to_findings(phase, note, chunker, chunk_size, chunk_overlap, store, results):
    row = (
        f"| {phase} | {note} | {chunker} | {chunk_size} | {chunk_overlap} | {store} "
        f"| {results['recall_at_1']:.2%} | {results['recall_at_3']:.2%} | {results['recall_at_5']:.2%} "
        f"| {results['mrr']:.4f} | {results['avg_top1_score']:.4f} "
        f"| {results['count']} | {results['avg_chars']} | {results['oversized_pct']}% "
        f"| {results['token_utilization_pct']}% "
        f"| {results['embed_time_ms']:.0f}ms | {results['avg_query_latency_ms']:.2f}ms |\n"
    )

    if not os.path.exists(FINDINGS_PATH):
        _init_findings()

    with open(FINDINGS_PATH, "a") as f:
        f.write(row)

    print(f"  → Appended to FINDINGS.md")


def _init_findings():
    header = """# RAG Retrieval Audit — Findings

Benchmark: 20 golden Q&A pairs across 4 academic papers.
Query fixed per phase. Only one pipeline layer changes per phase.
Baseline: Recursive chunker | chunk_size=800 | overlap=80 | FAISS

## Scorecard

| Phase | Note | Chunker | Size | Overlap | Store | R@1 | R@3 | R@5 | MRR | Top-1 | Chunks | Avg Size | Oversized | Token Util | Embed Time | Query Latency |
|-------|------|---------|------|---------|-------|-----|-----|-----|-----|-------|--------|----------|-----------|------------|------------|---------------|
"""
    with open(FINDINGS_PATH, "w") as f:
        f.write(header)


def main():
    parser = argparse.ArgumentParser(description="Evaluate one RAG pipeline phase.")
    parser.add_argument("--phase", required=True, help='e.g. "0-Baseline"')
    parser.add_argument("--chunker", default="recursive", choices=list(CHUNKERS.keys()))
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    parser.add_argument("--store", default="faiss", choices=list(STORES.keys()))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--note", default="", help="Short description of what changed")
    parser.add_argument("--save", action="store_true", help="Append results to FINDINGS.md")
    args = parser.parse_args()

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    chunker_fn = CHUNKERS[args.chunker]
    chunk_kwargs = {"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap}
    store_fn = STORES[args.store]

    print(f"\nRunning phase {args.phase}...")
    results = run_evaluation(chunker_fn, chunk_kwargs, store_fn, golden, args.top_k)

    print_results(args.phase, args.note, args.chunker, args.chunk_size, args.chunk_overlap, args.store, results)

    if args.save:
        append_to_findings(args.phase, args.note, args.chunker, args.chunk_size, args.chunk_overlap, args.store, results)


if __name__ == "__main__":
    main()
