"""
eval/run_phase6.py — Phase 6: query improvement (HyDE + RAG Fusion).

Builds on Phase 5 hybrid stack (BGE dense + BM25, weighted RRF α=0.7).
Tests 4 query strategies:
  1. baseline  — raw question → hybrid retrieve (Phase 5 best)
  2. hyde      — GPT fake answer → hybrid retrieve
  3. fusion    — 4 GPT query variants → hybrid retrieve each → RRF merge
  4. hyde+fusion — 4 variants each with HyDE → retrieve → RRF merge

Usage:
    python eval/run_phase6.py [--variants 4] [--save]
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
from openai import OpenAI
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
BGE_MODEL_ID     = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K   = 60
ALPHA   = 0.7   # dense weight in hybrid RRF
DENSE_K = 20
BM25_K  = 20
FINAL_K = 5


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def recall_at_k(hits, evidence, k):
    return 1.0 if any(evidence.lower() in h["text"].lower() for h in hits[:k]) else 0.0


def reciprocal_rank(hits, evidence, k):
    for i, h in enumerate(hits[:k]):
        if evidence.lower() in h["text"].lower():
            return 1.0 / (i + 1)
    return 0.0


def hybrid_retrieve(question, bi_encoder, store, bm25, chunks, final_k=FINAL_K, alpha=ALPHA):
    """Weighted RRF over dense + BM25, returns top-final_k chunks."""
    query_tokens = tokenize(question)
    qe = bi_encoder.encode([BGE_QUERY_PREFIX + question], normalize_embeddings=True)[0]

    dense_candidates = store.search(qe, k=DENSE_K)
    bm25_scores = bm25.get_scores(query_tokens)
    top_bm25_idx = list(np.argsort(bm25_scores)[::-1][:BM25_K])

    # Canonicalise by text
    canonical, ctr = {}, 0
    ranked_dense, ranked_bm25 = [], []
    for c in dense_candidates:
        if c["text"] not in canonical:
            canonical[c["text"]] = ctr; ctr += 1
        ranked_dense.append(canonical[c["text"]])
    for i in top_bm25_idx:
        t = chunks[i]["text"]
        if t not in canonical:
            canonical[t] = ctr; ctr += 1
        ranked_bm25.append(canonical[t])

    scores = {}
    for rank, cid in enumerate(ranked_dense):
        scores[cid] = scores.get(cid, 0.0) + alpha * (1.0 / (RRF_K + rank + 1))
    for rank, cid in enumerate(ranked_bm25):
        scores[cid] = scores.get(cid, 0.0) + (1 - alpha) * (1.0 / (RRF_K + rank + 1))

    inv = {v: k for k, v in canonical.items()}
    text_to_chunk = {c["text"]: c for c in dense_candidates}
    for i in top_bm25_idx:
        t = chunks[i]["text"]
        if t not in text_to_chunk:
            text_to_chunk[t] = chunks[i]

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [text_to_chunk[inv[cid]] for cid, _ in merged[:final_k]]


def rrf_merge_hits(ranked_hit_lists, k=RRF_K):
    """Merge multiple lists of chunk-dicts by text-keyed RRF."""
    scores = {}
    all_chunks = {}
    for hit_list in ranked_hit_lists:
        for rank, chunk in enumerate(hit_list):
            t = chunk["text"]
            scores[t] = scores.get(t, 0.0) + 1.0 / (k + rank + 1)
            all_chunks[t] = chunk
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [all_chunks[t] for t, _ in merged[:FINAL_K]]


def hyde_answer(client, question):
    """Ask GPT to write a short hypothetical passage that answers the question."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                "Write a short technical paragraph (3-5 sentences) from an academic paper "
                f"that directly answers this question:\n\n{question}\n\n"
                "Write only the passage, no preamble."
            )
        }],
        max_tokens=150,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def generate_variants(client, question, n=4):
    """Ask GPT to rephrase the question in n different ways."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Generate {n} different phrasings of this question for searching an academic paper. "
                f"Return only the questions, one per line, no numbering:\n\n{question}"
            )
        }],
        max_tokens=200,
        temperature=0.7,
    )
    lines = [l.strip() for l in resp.choices[0].message.content.strip().split("\n") if l.strip()]
    return lines[:n]


def evaluate(questions, retrieve_fn):
    r1s, r3s, r5s, mrrs, lats = [], [], [], [], []
    for item in questions:
        t0 = time.perf_counter()
        hits = retrieve_fn(item["question"])
        lats.append((time.perf_counter() - t0) * 1000)
        r1s.append(recall_at_k(hits, item["evidence"], 1))
        r3s.append(recall_at_k(hits, item["evidence"], 3))
        r5s.append(recall_at_k(hits, item["evidence"], 5))
        mrrs.append(reciprocal_rank(hits, item["evidence"], FINAL_K))
    return {
        "r1": np.mean(r1s), "r3": np.mean(r3s),
        "r5": np.mean(r5s), "mrr": np.mean(mrrs),
        "lat": np.mean(lats),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set"); sys.exit(1)
    client = OpenAI(api_key=api_key)

    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    print("Loading chunks...")
    pages  = load_all_pdfs(PAPERS_DIR)
    chunks = section_wise_chunk(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"  {len(chunks)} chunks")

    print("Building BM25 index...")
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    print("Building BGE+FAISS index...")
    bi_encoder = SentenceTransformer(BGE_MODEL_ID)
    doc_embs = bi_encoder.encode([c["text"] for c in chunks], batch_size=64, normalize_embeddings=True)
    store = FaissStore(dimension=doc_embs.shape[1])
    store.add(chunks, doc_embs)
    print(f"  {store.count} vectors indexed")

    # --- Strategy 1: baseline hybrid (Phase 5) ---
    print("\n[1/4] Baseline hybrid (α=0.7)...")
    baseline_res = evaluate(golden, lambda q: hybrid_retrieve(q, bi_encoder, store, bm25, chunks))

    # --- Strategy 2: HyDE ---
    print("[2/4] HyDE — generating hypothetical answers...")
    hyde_cache = {}
    for item in golden:
        hyde_cache[item["question"]] = hyde_answer(client, item["question"])
        print(f"  Q: {item['question'][:60]}...")
        print(f"  A: {hyde_cache[item['question']][:80]}...\n")

    def retrieve_hyde(question):
        fake_answer = hyde_cache[question]
        return hybrid_retrieve(fake_answer, bi_encoder, store, bm25, chunks)

    hyde_res = evaluate(golden, retrieve_hyde)

    # --- Strategy 3: RAG Fusion ---
    print(f"[3/4] RAG Fusion — generating {args.variants} query variants per question...")
    fusion_cache = {}
    for item in golden:
        variants = generate_variants(client, item["question"], n=args.variants)
        fusion_cache[item["question"]] = variants
        print(f"  Q: {item['question'][:55]}")
        for v in variants:
            print(f"    → {v}")

    def retrieve_fusion(question):
        variants = fusion_cache[question]
        hit_lists = [hybrid_retrieve(v, bi_encoder, store, bm25, chunks, final_k=20) for v in variants]
        return rrf_merge_hits(hit_lists)

    fusion_res = evaluate(golden, retrieve_fusion)

    # --- Strategy 4: HyDE + Fusion ---
    print(f"[4/4] HyDE + Fusion — HyDE on each variant...")

    def retrieve_hyde_fusion(question):
        variants = fusion_cache[question]
        hit_lists = []
        for v in variants:
            fake = hyde_answer(client, v)
            hit_lists.append(hybrid_retrieve(fake, bi_encoder, store, bm25, chunks, final_k=20))
        return rrf_merge_hits(hit_lists)

    hyde_fusion_res = evaluate(golden, retrieve_hyde_fusion)

    # --- Print results ---
    strategies = {
        "Hybrid (Ph5)":   baseline_res,
        "HyDE":           hyde_res,
        "RAG Fusion":     fusion_res,
        "HyDE + Fusion":  hyde_fusion_res,
    }

    print(f"\n{'Metric':<14}", end="")
    for name in strategies:
        print(f" {name:>14}", end="")
    print()
    print("─" * (14 + 15 * len(strategies)))

    for metric, label in [("r1","Recall@1"),("r3","Recall@3"),("r5","Recall@5"),("mrr","MRR"),("lat","Latency")]:
        print(f"{label:<14}", end="")
        for res in strategies.values():
            v = res[metric]
            if metric == "lat":
                print(f" {v:>13.1f}ms", end="")
            elif metric == "mrr":
                print(f" {v:>14.4f}", end="")
            else:
                print(f" {v:>13.0%}", end="")
        print()

    if args.save:
        best = max(strategies.items(), key=lambda x: (x[1]["r5"], x[1]["mrr"]))
        with open(FINDINGS_PATH, "a") as f:
            f.write("\n\n## Phase 6 — Query Improvement (HyDE + RAG Fusion) Raw Results\n\n")
            f.write(f"Base retrieval: hybrid RRF α={ALPHA}, BM25-k={BM25_K}, dense-k={DENSE_K}, final-k={FINAL_K}\n")
            f.write(f"GPT model: gpt-4o-mini | RAG Fusion variants: {args.variants}\n\n")
            f.write(f"| Metric | Hybrid (Ph5) | HyDE | RAG Fusion | HyDE+Fusion |\n")
            f.write(f"|--------|-------------|------|------------|-------------|\n")
            for metric, label in [("r1","Recall@1"),("r3","Recall@3"),("r5","Recall@5"),("mrr","MRR")]:
                fmt = ".4f" if metric == "mrr" else ".0%"
                row = f"| {label} |"
                for res in strategies.values():
                    row += f" {res[metric]:{fmt}} |"
                f.write(row + "\n")
            f.write(f"\n**Best:** {best[0]} → R@5={best[1]['r5']:.0%}, MRR={best[1]['mrr']:.4f}\n")
        print("\n  → Saved to FINDINGS.md")


if __name__ == "__main__":
    main()
