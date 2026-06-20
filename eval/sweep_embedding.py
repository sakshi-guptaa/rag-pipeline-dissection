"""
eval/sweep_embedding.py — Phase 3 embedding model sweep.

Fixed config: section_wise chunker, size=1000, overlap=100, FAISS.
Tests 4 models, each loaded once and evaluated against all 20 golden questions.

Usage:
    python eval/sweep_embedding.py [--save]

Model-specific notes:
  BGE  — queries need prefix "Represent this sentence for searching relevant passages: "
  Nomic — queries need prefix "search_query: ", docs need "search_document: "
          requires trust_remote_code=True
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.loader import load_all_pdfs
from chunking.section_wise import chunk as section_wise_chunk
from vectordb.faiss_store import FaissStore

PAPERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")
GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json")
FINDINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FINDINGS.md")

CHUNK_SIZE   = 1000
CHUNK_OVERLAP = 100

MODELS = [
    {
        "name": "all-mpnet-base-v2",
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "query_prefix": "",
        "doc_prefix": "",
        "trust_remote_code": False,
        "note": "Baseline",
    },
    {
        "name": "multi-qa-mpnet-base-dot-v1",
        "model_id": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "query_prefix": "",
        "doc_prefix": "",
        "trust_remote_code": False,
        "note": "Q&A-optimized, same arch as baseline",
    },
    {
        "name": "bge-base-en-v1.5",
        "model_id": "BAAI/bge-base-en-v1.5",
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "doc_prefix": "",
        "trust_remote_code": False,
        "note": "Top MTEB retrieval, needs query prefix",
    },
    {
        "name": "nomic-embed-text-v1",
        "model_id": "nomic-ai/nomic-embed-text-v1",
        "query_prefix": "search_query: ",
        "doc_prefix": "search_document: ",
        "trust_remote_code": True,
        "note": "8192-token limit, eliminates truncation",
    },
]


def recall_at_k(hits, evidence, k):
    return 1.0 if any(evidence.lower() in h["text"].lower() for h in hits[:k]) else 0.0


def reciprocal_rank(hits, evidence, k):
    for i, h in enumerate(hits[:k]):
        if evidence.lower() in h["text"].lower():
            return 1.0 / (i + 1)
    return 0.0


def embed(model, texts, prefix="", batch_size=64):
    if prefix:
        texts = [prefix + t for t in texts]
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True)


def evaluate_model(cfg, chunks, golden):
    print(f"\n  Loading {cfg['name']}...", flush=True)
    model = SentenceTransformer(
        cfg["model_id"],
        trust_remote_code=cfg["trust_remote_code"],
    )
    token_limit = model.max_seq_length
    print(f"  Token limit: {token_limit} | Dim: {model.get_sentence_embedding_dimension()}")

    texts = [c["text"] for c in chunks]

    t0 = time.perf_counter()
    doc_embeddings = embed(model, texts, prefix=cfg["doc_prefix"])
    embed_time = (time.perf_counter() - t0) * 1000

    dim = doc_embeddings.shape[1]
    store = FaissStore(dimension=dim)
    store.add(chunks, doc_embeddings)

    r1s, r3s, r5s, mrrs, top1_scores, latencies = [], [], [], [], [], []

    for item in golden:
        qe = embed(model, [item["question"]], prefix=cfg["query_prefix"])[0]

        t0 = time.perf_counter()
        hits = store.search(qe, k=5)
        latencies.append((time.perf_counter() - t0) * 1000)

        r1s.append(recall_at_k(hits, item["evidence"], 1))
        r3s.append(recall_at_k(hits, item["evidence"], 3))
        r5s.append(recall_at_k(hits, item["evidence"], 5))
        mrrs.append(reciprocal_rank(hits, item["evidence"], 5))
        if hits:
            top1_scores.append(hits[0]["score"])

    # token utilization
    tokenizer = model.tokenizer
    token_counts = [len(tokenizer.tokenize(t)) for t in texts]
    oversized = sum(1 for t in token_counts if t > token_limit)

    return {
        "name": cfg["name"],
        "note": cfg["note"],
        "token_limit": token_limit,
        "dim": dim,
        "recall_at_1": round(np.mean(r1s), 4),
        "recall_at_3": round(np.mean(r3s), 4),
        "recall_at_5": round(np.mean(r5s), 4),
        "mrr": round(np.mean(mrrs), 4),
        "avg_top1_score": round(np.mean(top1_scores), 4),
        "oversized_count": oversized,
        "oversized_pct": round(oversized / len(chunks) * 100, 1),
        "avg_token_util_pct": round(np.mean([min(t / token_limit, 1.0) for t in token_counts]) * 100, 1),
        "embed_time_ms": round(embed_time, 0),
        "avg_latency_ms": round(np.mean(latencies), 2),
    }


def print_table(results):
    print(f"\n{'Model':<30} {'Lim':>4} {'Dim':>5} | {'R@1':>5} {'R@3':>5} {'R@5':>5} {'MRR':>7} {'Top1':>6} | {'Ovsz':>5} {'TkUt':>5} | {'EmbMs':>7} {'Lat':>6}")
    print("─" * 105)
    best_r5 = max(r["recall_at_5"] for r in results)
    for r in results:
        marker = " ◀ BEST" if r["recall_at_5"] == best_r5 else ""
        print(
            f"{r['name']:<30} {r['token_limit']:>4} {r['dim']:>5} | "
            f"{r['recall_at_1']:>4.0%} {r['recall_at_3']:>4.0%} {r['recall_at_5']:>4.0%} {r['mrr']:>7.4f} {r['avg_top1_score']:>6.4f} | "
            f"{r['oversized_pct']:>4.1f}% {r['avg_token_util_pct']:>4.1f}% | "
            f"{r['embed_time_ms']:>6.0f}ms {r['avg_latency_ms']:>5.2f}ms{marker}"
        )


def save_to_findings(results):
    rows = ""
    for r in results:
        rows += (
            f"| {r['name']} | {r['note']} | {r['token_limit']} | {r['dim']} "
            f"| {r['recall_at_1']:.0%} | {r['recall_at_3']:.0%} | {r['recall_at_5']:.0%} "
            f"| {r['mrr']:.4f} | {r['avg_top1_score']:.4f} "
            f"| {r['oversized_pct']}% | {r['avg_token_util_pct']}% "
            f"| {r['embed_time_ms']:.0f}ms | {r['avg_latency_ms']:.2f}ms |\n"
        )
    best = max(results, key=lambda r: (r["recall_at_5"], r["mrr"]))
    with open(FINDINGS_PATH, "a") as f:
        f.write("\n\n## Phase 3 — Embedding Sweep Raw Results\n\n")
        f.write("| Model | Note | Token Limit | Dim | R@1 | R@3 | R@5 | MRR | Top-1 | Oversized | Token Util | Embed Time | Query Latency |\n")
        f.write("|-------|------|------------|-----|-----|-----|-----|-----|-------|-----------|------------|------------|---------------|\n")
        f.write(rows)
        f.write(f"\n**Winner:** `{best['name']}` → R@5={best['recall_at_5']:.0%}, MRR={best['mrr']:.4f}\n")
    print(f"\n  → Saved to FINDINGS.md")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    print("Loading pages and chunking (fixed: section_wise size=1000 overlap=100)...")
    pages = load_all_pdfs(PAPERS_DIR)
    chunks = section_wise_chunk(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"  {len(chunks)} chunks from {len(pages)} documents\n")

    results = []
    for i, cfg in enumerate(MODELS):
        print(f"[{i+1}/{len(MODELS)}] {cfg['name']} ({cfg['note']})")
        try:
            r = evaluate_model(cfg, chunks, golden)
            results.append(r)
            print(f"  R@1={r['recall_at_1']:.0%}  R@3={r['recall_at_3']:.0%}  R@5={r['recall_at_5']:.0%}  MRR={r['mrr']:.4f}  Top1={r['avg_top1_score']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print_table(results)

    if args.save:
        best = save_to_findings(results)
        print(f"\nBest: {best['name']} → R@5={best['recall_at_5']:.0%}, MRR={best['mrr']:.4f}")


if __name__ == "__main__":
    main()
