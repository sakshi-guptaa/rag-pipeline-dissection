"""
eval/run_phase5.py — Phase 5: BM25 + dense hybrid search with RRF.

Two retrieval signals merged via Reciprocal Rank Fusion:
  - BM25 (keyword): exact term matching, no embeddings needed
  - BGE dense (semantic): same as Phase 3

RRF score = sum(1 / (k + rank_i)) across all retrievers, where k=60.
Higher RRF score = better combined rank.

Fixed config: section_wise chunker, size=1000, overlap=100, FAISS for dense.

Usage:
    python eval/run_phase5.py [--bm25-k INT] [--dense-k INT] [--final-k INT] [--save]

Defaults: bm25-k=20, dense-k=20, final-k=5 (matches the other paper's setup)
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.loader import load_all_pdfs
from chunking.section_wise import chunk as section_wise_chunk
from vectordb.faiss_store import FaissStore

PAPERS_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")
GOLDEN_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json")
FINDINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FINDINGS.md")

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 100
RRF_K         = 60   # standard RRF constant

BGE_MODEL_ID     = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def rrf_merge(dense_ranked, bm25_ranked, k=RRF_K, alpha=0.5):
    """Weighted RRF: alpha controls dense weight, (1-alpha) controls BM25 weight."""
    scores = {}
    for rank, doc_id in enumerate(dense_ranked):
        scores[doc_id] = scores.get(doc_id, 0.0) + alpha * (1.0 / (k + rank + 1))
    for rank, doc_id in enumerate(bm25_ranked):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1 - alpha) * (1.0 / (k + rank + 1))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def recall_at_k(hits, evidence, k):
    return 1.0 if any(evidence.lower() in h["text"].lower() for h in hits[:k]) else 0.0


def reciprocal_rank(hits, evidence, k):
    for i, h in enumerate(hits[:k]):
        if evidence.lower() in h["text"].lower():
            return 1.0 / (i + 1)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25-k",  type=int,   default=20,  dest="bm25_k")
    parser.add_argument("--dense-k", type=int,   default=20,  dest="dense_k")
    parser.add_argument("--final-k", type=int,   default=5,   dest="final_k")
    parser.add_argument("--alphas",  type=float, nargs="+",
                        default=[0.5, 0.6, 0.7, 0.8],
                        help="Dense weight(s) for weighted RRF (0=BM25 only, 1=dense only)")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    print("Loading pages and chunking (section_wise size=1000 overlap=100)...")
    pages  = load_all_pdfs(PAPERS_DIR)
    chunks = section_wise_chunk(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"  {len(chunks)} chunks from {len(pages)} documents")

    # BM25 index
    print("\nBuilding BM25 index...")
    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"  {len(chunks)} documents indexed")

    # Dense index
    print(f"\nLoading BGE and building FAISS index...")
    bi_encoder = SentenceTransformer(BGE_MODEL_ID)
    texts = [c["text"] for c in chunks]
    t0 = time.perf_counter()
    doc_embeddings = bi_encoder.encode(texts, batch_size=64, normalize_embeddings=True)
    embed_time_ms = (time.perf_counter() - t0) * 1000
    store = FaissStore(dimension=doc_embeddings.shape[1])
    store.add(chunks, doc_embeddings)
    print(f"  {store.count} vectors indexed  embed={embed_time_ms:.0f}ms")

    # Pre-compute per-question retrieval signals (shared across all alpha values)
    print(f"\nPre-computing retrieval signals for {len(golden)} questions...")

    r1_dense, r3_dense, r5_dense, mrr_dense, lat_dense = [], [], [], [], []
    r1_bm25,  r3_bm25,  r5_bm25,  mrr_bm25,  lat_bm25  = [], [], [], [], []

    per_question = []  # stores (dense_canon_ranked, bm25_canon_ranked, text_to_chunk) per question

    for item in golden:
        query_tokens = tokenize(item["question"])
        query_embed  = bi_encoder.encode(
            [BGE_QUERY_PREFIX + item["question"]], normalize_embeddings=True
        )[0]

        # Dense-only
        t0 = time.perf_counter()
        dense_hits = store.search(query_embed, k=args.final_k)
        lat_dense.append((time.perf_counter() - t0) * 1000)
        r1_dense.append(recall_at_k(dense_hits, item["evidence"], 1))
        r3_dense.append(recall_at_k(dense_hits, item["evidence"], 3))
        r5_dense.append(recall_at_k(dense_hits, item["evidence"], 5))
        mrr_dense.append(reciprocal_rank(dense_hits, item["evidence"], args.final_k))

        # BM25-only
        t0 = time.perf_counter()
        bm25_scores = bm25.get_scores(query_tokens)
        lat_bm25.append((time.perf_counter() - t0) * 1000)
        top_bm25_idx = list(np.argsort(bm25_scores)[::-1][:args.bm25_k])
        bm25_hits = [chunks[i] for i in top_bm25_idx[:args.final_k]]
        r1_bm25.append(recall_at_k(bm25_hits, item["evidence"], 1))
        r3_bm25.append(recall_at_k(bm25_hits, item["evidence"], 3))
        r5_bm25.append(recall_at_k(bm25_hits, item["evidence"], 5))
        mrr_bm25.append(reciprocal_rank(bm25_hits, item["evidence"], args.final_k))

        # Build canonical ranked lists for RRF (shared across alphas)
        dense_candidates = store.search(query_embed, k=args.dense_k)
        canonical = {}
        ctr = 0
        ranked_dense_canon, ranked_bm25_canon = [], []

        for c in dense_candidates:
            t = c["text"]
            if t not in canonical:
                canonical[t] = ctr; ctr += 1
            ranked_dense_canon.append(canonical[t])

        for idx in top_bm25_idx:
            t = chunks[idx]["text"]
            if t not in canonical:
                canonical[t] = ctr; ctr += 1
            ranked_bm25_canon.append(canonical[t])

        inv_canonical = {v: k for k, v in canonical.items()}
        text_to_chunk = {c["text"]: c for c in dense_candidates}
        for idx in top_bm25_idx:
            t = chunks[idx]["text"]
            if t not in text_to_chunk:
                text_to_chunk[t] = chunks[idx]

        per_question.append({
            "evidence": item["evidence"],
            "dense_canon": ranked_dense_canon,
            "bm25_canon":  ranked_bm25_canon,
            "inv_canonical": inv_canonical,
            "text_to_chunk": text_to_chunk,
        })

    # Print baselines
    dense_res = (np.mean(r1_dense), np.mean(r3_dense), np.mean(r5_dense),
                 np.mean(mrr_dense), np.mean(lat_dense))
    bm25_res  = (np.mean(r1_bm25),  np.mean(r3_bm25),  np.mean(r5_bm25),
                 np.mean(mrr_bm25),  np.mean(lat_bm25))

    # Sweep alphas
    alpha_results = {}
    for alpha in args.alphas:
        r1s, r3s, r5s, mrrs = [], [], [], []
        for pq in per_question:
            merged = rrf_merge(pq["dense_canon"], pq["bm25_canon"], k=RRF_K, alpha=alpha)
            hits = [pq["text_to_chunk"][pq["inv_canonical"][cid]]
                    for cid, _ in merged[:args.final_k]]
            r1s.append(recall_at_k(hits, pq["evidence"], 1))
            r3s.append(recall_at_k(hits, pq["evidence"], 3))
            r5s.append(recall_at_k(hits, pq["evidence"], 5))
            mrrs.append(reciprocal_rank(hits, pq["evidence"], args.final_k))
        alpha_results[alpha] = (np.mean(r1s), np.mean(r3s), np.mean(r5s), np.mean(mrrs))

    # Print table
    alpha_labels = [f"α={a}" for a in args.alphas]
    header = f"{'Metric':<14} {'Dense':>8} {'BM25':>8}" + "".join(f" {l:>10}" for l in alpha_labels)
    print(f"\n{header}")
    print("─" * (32 + 11 * len(args.alphas)))

    for i, label in enumerate(["Recall@1", "Recall@3", "Recall@5", "MRR"]):
        d = dense_res[i]
        b = bm25_res[i]
        if label == "MRR":
            row = f"{label:<14} {d:>8.4f} {b:>8.4f}"
            for alpha in args.alphas:
                row += f" {alpha_results[alpha][i]:>10.4f}"
        else:
            row = f"{label:<14} {d:>7.0%} {b:>8.0%}"
            for alpha in args.alphas:
                row += f" {alpha_results[alpha][i]:>10.0%}"
        print(row)

    print(f"\n  (α=0.5 = equal weight, α=1.0 = dense only, α=0.0 = BM25 only)")

    best_alpha = max(args.alphas, key=lambda a: (alpha_results[a][2], alpha_results[a][3]))
    print(f"  Best alpha by R@5+MRR: α={best_alpha}")

    if args.save:
        d, b = dense_res, bm25_res
        with open(FINDINGS_PATH, "a") as f:
            f.write("\n\n## Phase 5 — Hybrid Search Alpha Sweep\n\n")
            f.write(f"Config: BM25 top-{args.bm25_k} + BGE dense top-{args.dense_k} → weighted RRF(k={RRF_K}) → top-{args.final_k}\n\n")
            f.write(f"| Metric | Dense (Ph3) | BM25-only |")
            for alpha in args.alphas:
                f.write(f" α={alpha} |")
            f.write("\n|--------|------------|-----------|")
            for _ in args.alphas:
                f.write("--------|")
            f.write("\n")
            for i, label in enumerate(["Recall@1", "Recall@3", "Recall@5", "MRR"]):
                fmt = ".4f" if label == "MRR" else ".0%"
                row = f"| {label} | {d[i]:{fmt}} | {b[i]:{fmt}} |"
                for alpha in args.alphas:
                    row += f" {alpha_results[alpha][i]:{fmt}} |"
                f.write(row + "\n")
            f.write(f"\n**Best alpha:** α={best_alpha} → R@5={alpha_results[best_alpha][2]:.0%}, MRR={alpha_results[best_alpha][3]:.4f}\n")
        print("\n  → Saved to FINDINGS.md")


if __name__ == "__main__":
    main()
