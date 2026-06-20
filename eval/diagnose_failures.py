"""
eval/diagnose_failures.py — Inspect the questions that still fail after all phases.

For each question in the golden set, checks:
  1. Is the evidence string present in ANY chunk? (parser/chunking problem)
  2. What rank does the correct chunk get in dense-only retrieval? (top-50)
  3. What rank does the correct chunk get in BM25-only? (top-50)
  4. What rank does the correct chunk get in hybrid RRF? (top-50)
  5. Shows the top-3 retrieved chunks vs the correct evidence

Usage:
    python eval/diagnose_failures.py
"""

import json
import os
import re
import sys

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.loader import load_all_pdfs
from chunking.section_wise import chunk as section_wise_chunk
from vectordb.faiss_store import FaissStore

PAPERS_DIR   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "papers")
GOLDEN_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json")

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 100
BGE_MODEL_ID     = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K  = 60
ALPHA  = 0.7
SEARCH_K = 50   # search wide to find where the correct chunk actually ranks


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def find_rank(hits, evidence):
    for i, h in enumerate(hits):
        if evidence.lower() in h["text"].lower():
            return i + 1
    return None  # not found in top-K


def hybrid_search(question, bi_encoder, store, bm25, chunks, k=SEARCH_K):
    query_tokens = tokenize(question)
    qe = bi_encoder.encode([BGE_QUERY_PREFIX + question], normalize_embeddings=True)[0]

    dense_hits = store.search(qe, k=k)
    bm25_scores = bm25.get_scores(query_tokens)
    top_bm25_idx = list(np.argsort(bm25_scores)[::-1][:k])

    canonical, ctr = {}, 0
    ranked_dense, ranked_bm25 = [], []
    for c in dense_hits:
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
        scores[cid] = scores.get(cid, 0.0) + ALPHA * (1.0 / (RRF_K + rank + 1))
    for rank, cid in enumerate(ranked_bm25):
        scores[cid] = scores.get(cid, 0.0) + (1 - ALPHA) * (1.0 / (RRF_K + rank + 1))

    inv = {v: k for k, v in canonical.items()}
    text_to_chunk = {c["text"]: c for c in dense_hits}
    for i in top_bm25_idx:
        t = chunks[i]["text"]
        if t not in text_to_chunk:
            text_to_chunk[t] = chunks[i]

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [text_to_chunk[inv[cid]] for cid, _ in merged if inv[cid] in text_to_chunk]


def main():
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    print("Loading chunks...")
    pages  = load_all_pdfs(PAPERS_DIR)
    chunks = section_wise_chunk(pages, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"  {len(chunks)} chunks\n")

    print("Building indexes...")
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    bi_encoder = SentenceTransformer(BGE_MODEL_ID)
    doc_embs = bi_encoder.encode([c["text"] for c in chunks], batch_size=64, normalize_embeddings=True)
    store = FaissStore(dimension=doc_embs.shape[1])
    store.add(chunks, doc_embs)
    print(f"  Ready\n")

    print("=" * 80)
    print("QUESTION-LEVEL DIAGNOSIS (searching top-50)")
    print("=" * 80)

    failing_hybrid = []

    for i, item in enumerate(golden):
        q = item["question"]
        ev = item["evidence"]

        # Check if evidence exists in ANY chunk
        containing = [c for c in chunks if ev.lower() in c["text"].lower()]
        in_index = len(containing) > 0

        # Dense rank
        dense_hits = store.search(
            bi_encoder.encode([BGE_QUERY_PREFIX + q], normalize_embeddings=True)[0], k=SEARCH_K
        )
        dense_rank = find_rank(dense_hits, ev)

        # BM25 rank
        bm25_scores = bm25.get_scores(tokenize(q))
        top_bm25 = [{"text": chunks[j]["text"]} for j in np.argsort(bm25_scores)[::-1][:SEARCH_K]]
        bm25_rank = find_rank(top_bm25, ev)

        # Hybrid rank
        hybrid_hits = hybrid_search(q, bi_encoder, store, bm25, chunks)
        hybrid_rank = find_rank(hybrid_hits, ev)

        hit5 = hybrid_rank is not None and hybrid_rank <= 5
        status = "✓" if hit5 else "✗ FAIL"

        print(f"\n[{i+1:02d}] {status}  Q: {q}")
        print(f"       In index: {'YES (' + str(len(containing)) + ' chunk(s))' if in_index else '❌ NOT IN ANY CHUNK'}")
        print(f"       Dense rank:  {dense_rank  if dense_rank  else f'>{ SEARCH_K}'}")
        print(f"       BM25  rank:  {bm25_rank   if bm25_rank   else f'>{ SEARCH_K}'}")
        print(f"       Hybrid rank: {hybrid_rank if hybrid_rank else f'>{ SEARCH_K}'}")

        if not hit5:
            failing_hybrid.append(item)
            print(f"\n       EVIDENCE snippet: \"{ev[:120]}...\"")
            if in_index:
                print(f"\n       Containing chunk (first 300 chars):")
                print(f"       {containing[0]['text'][:300].replace(chr(10), ' ')}...")
            print(f"\n       Top-3 hybrid results:")
            for rank, h in enumerate(hybrid_hits[:3], 1):
                print(f"         [{rank}] {h['text'][:150].replace(chr(10), ' ')}...")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {len(golden) - len(failing_hybrid)}/{len(golden)} pass hybrid@5")
    print(f"         {len(failing_hybrid)} still failing:")
    for item in failing_hybrid:
        print(f"  - {item['question']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
