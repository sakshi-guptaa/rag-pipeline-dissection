"""
eval/run_phase4.py — Phase 4: cross-encoder reranker.

Two-stage retrieval:
  Stage 1: BGE bi-encoder retrieves top-CANDIDATE_K (20) from FAISS
  Stage 2: cross-encoder/ms-marco-MiniLM-L-6-v2 reranks → return top-5

Fixed config: section_wise chunker, size=1000, overlap=100, FAISS, BGE embeddings.

Usage:
    python eval/run_phase4.py [--save]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.loader import load_all_pdfs
from chunking.section_wise import chunk as section_wise_chunk
from vectordb.faiss_store import FaissStore

PAPERS_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")
GOLDEN_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json")
FINDINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FINDINGS.md")

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 100
CANDIDATE_K   = 20   # bi-encoder retrieves this many before reranking
FINAL_K       = 5    # cross-encoder returns this many

BGE_MODEL_ID      = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX  = "Represent this sentence for searching relevant passages: "
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def recall_at_k(hits, evidence, k):
    return 1.0 if any(evidence.lower() in h["text"].lower() for h in hits[:k]) else 0.0


def reciprocal_rank(hits, evidence, k):
    for i, h in enumerate(hits[:k]):
        if evidence.lower() in h["text"].lower():
            return 1.0 / (i + 1)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    print("Loading pages and chunking (section_wise size=1000 overlap=100)...")
    pages  = load_all_pdfs(PAPERS_DIR)
    chunks = section_wise_chunk(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"  {len(chunks)} chunks from {len(pages)} documents")

    print(f"\nLoading bi-encoder: {BGE_MODEL_ID}...")
    bi_encoder = SentenceTransformer(BGE_MODEL_ID)

    texts = [c["text"] for c in chunks]
    t0 = time.perf_counter()
    doc_embeddings = bi_encoder.encode(texts, batch_size=64, normalize_embeddings=True)
    embed_time_ms = (time.perf_counter() - t0) * 1000

    store = FaissStore(dimension=doc_embeddings.shape[1])
    store.add(chunks, doc_embeddings)
    print(f"  {store.count} vectors indexed  embed={embed_time_ms:.0f}ms")

    print(f"\nLoading cross-encoder: {RERANKER_MODEL_ID}...")
    cross_encoder = CrossEncoder(RERANKER_MODEL_ID)
    print("  Ready")

    r1s, r3s, r5s, mrrs, top1_scores = [], [], [], [], []
    bi_latencies, rerank_latencies = [], []

    for item in golden:
        query = BGE_QUERY_PREFIX + item["question"]
        qe = bi_encoder.encode([query], normalize_embeddings=True)[0]

        # Stage 1: bi-encoder retrieves CANDIDATE_K
        t0 = time.perf_counter()
        candidates = store.search(qe, k=CANDIDATE_K)
        bi_latencies.append((time.perf_counter() - t0) * 1000)

        # Stage 2: cross-encoder reranks
        pairs = [[item["question"], c["text"]] for c in candidates]
        t0 = time.perf_counter()
        scores = cross_encoder.predict(pairs)
        rerank_latencies.append((time.perf_counter() - t0) * 1000)

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        hits = [c for _, c in ranked[:FINAL_K]]

        r1s.append(recall_at_k(hits, item["evidence"], 1))
        r3s.append(recall_at_k(hits, item["evidence"], 3))
        r5s.append(recall_at_k(hits, item["evidence"], 5))
        mrrs.append(reciprocal_rank(hits, item["evidence"], 5))
        if hits:
            top1_scores.append(hits[0]["score"])

    r1  = round(np.mean(r1s),  4)
    r3  = round(np.mean(r3s),  4)
    r5  = round(np.mean(r5s),  4)
    mrr = round(np.mean(mrrs), 4)
    top1 = round(np.mean(top1_scores), 4)
    avg_bi      = round(np.mean(bi_latencies),     2)
    avg_rerank  = round(np.mean(rerank_latencies), 2)
    avg_total   = round(avg_bi + avg_rerank,        2)

    print(f"\n{'Metric':<20} {'No reranker (Ph3)':>20} {'+ Reranker (Ph4)':>20}")
    print("─" * 62)
    print(f"{'Recall@1':<20} {'60%':>20} {r1:>19.0%}")
    print(f"{'Recall@3':<20} {'80%':>20} {r3:>19.0%}")
    print(f"{'Recall@5':<20} {'85%':>20} {r5:>19.0%}")
    print(f"{'MRR':<20} {'0.6933':>20} {mrr:>20.4f}")
    print(f"{'Avg Top-1 Score':<20} {'0.7036':>20} {top1:>20.4f}")
    print(f"{'Bi-encoder latency':<20} {'0.08ms':>20} {avg_bi:>18.2f}ms")
    print(f"{'Reranker latency':<20} {'—':>20} {avg_rerank:>18.2f}ms")
    print(f"{'Total latency':<20} {'0.08ms':>20} {avg_total:>18.2f}ms")

    if args.save:
        with open(FINDINGS_PATH, "a") as f:
            f.write("\n\n## Phase 4 — Retrieval + Reranker Raw Results\n\n")
            f.write(f"Cross-encoder: `{RERANKER_MODEL_ID}` · Stage 1: top-{CANDIDATE_K} bi-encoder · Stage 2: rerank → top-{FINAL_K}\n\n")
            f.write("| Metric | Phase 3 (no reranker) | Phase 4 (+ reranker) | Delta |\n")
            f.write("|--------|----------------------|----------------------|-------|\n")
            f.write(f"| Recall@1 | 60% | {r1:.0%} | {r1-0.60:+.0%} |\n")
            f.write(f"| Recall@3 | 80% | {r3:.0%} | {r3-0.80:+.0%} |\n")
            f.write(f"| Recall@5 | 85% | {r5:.0%} | {r5-0.85:+.0%} |\n")
            f.write(f"| MRR | 0.6933 | {mrr:.4f} | {mrr-0.6933:+.4f} |\n")
            f.write(f"| Avg Top-1 Score | 0.7036 | {top1:.4f} | {top1-0.7036:+.4f} |\n")
            f.write(f"| Bi-encoder latency | 0.08ms | {avg_bi:.2f}ms | — |\n")
            f.write(f"| Reranker latency | — | {avg_rerank:.2f}ms | — |\n")
            f.write(f"| Total query latency | 0.08ms | {avg_total:.2f}ms | — |\n")
        print("\n  → Saved to FINDINGS.md")


if __name__ == "__main__":
    main()
