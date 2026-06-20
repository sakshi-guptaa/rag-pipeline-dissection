# RAG Pipeline Dissection — Findings

A systematic, phase-by-phase audit of what actually moves the needle in RAG retrieval quality.
Each phase changes exactly one layer of the pipeline and measures the impact on a fixed benchmark.

---

## Benchmark Setup

| Parameter | Value |
|-----------|-------|
| **Corpus** | 4 academic papers (Attention Is All You Need, RAG original, RAGAS, HNSW) |
| **Golden set** | 20 hand-crafted question–evidence pairs (`eval/golden_set.json`) |
| **Embedding model** | `sentence-transformers/all-mpnet-base-v2` (768-dim, 384-token limit) |
| **Baseline chunker** | Recursive, chunk_size=800, chunk_overlap=80 |
| **Baseline store** | FAISS (brute-force exact search, IndexFlatIP) |
| **Top-K** | 5 |

---

## Metrics Definitions

### Retrieval Quality

| Metric | Definition | Range | Ideal |
|--------|-----------|--------|-------|
| **Recall@1** | Fraction of questions where the correct evidence appears in the single top result | 0–1 | 1.0 |
| **Recall@3** | Fraction of questions where correct evidence appears anywhere in top 3 results | 0–1 | 1.0 |
| **Recall@5** | Fraction of questions where correct evidence appears anywhere in top 5 results | 0–1 | 1.0 |
| **MRR** (Mean Reciprocal Rank) | Average of 1/rank of the first correct result across all questions. MRR=1.0 means always rank 1; MRR=0.5 means always rank 2 | 0–1 | 1.0 |
| **Avg Top-1 Score** | Mean cosine similarity between the query and the best-matching chunk, averaged across all 20 questions | 0–1 | 1.0 |

> **Recall@5 and MRR are the headline metrics.** Recall@5 tells you whether the answer is retrievable at all. MRR tells you how reliably it lands at the top.

### Chunk Quality

| Metric | Definition | Why It Matters |
|--------|-----------|----------------|
| **Total chunks** | Number of chunks produced from the full corpus | Too few = coarse coverage; too many = fragmented context |
| **Avg size (chars)** | Mean character length per chunk | Proxy for information density per chunk |
| **Std dev (chars)** | Standard deviation of chunk sizes | High std dev = inconsistent chunking |
| **Oversized (%)** | % of chunks exceeding 384 tokens (embedding model limit) | Oversized chunks are silently truncated — tail content disappears from vector space |
| **Token utilization (%)** | Average % of the 384-token window actually used per chunk | <30% = under-packing (wasted capacity); >90% = risk of truncation |

### Operational

| Metric | Definition |
|--------|-----------|
| **Parse time** | Time to read and extract text from all PDFs |
| **Embed time** | Time to encode all chunks into vectors |
| **Index time** | Time to load vectors into the vector store |
| **Avg query latency** | Mean time for a single nearest-neighbour search |

---

## Scorecard

| Phase | Change | R@1 | R@3 | R@5 | MRR | Top-1 Score | Chunks | Avg Size | Oversized | Token Util | Embed Time | Query Latency |
|-------|--------|-----|-----|-----|-----|-------------|--------|----------|-----------|------------|------------|---------------|
| 0 — Baseline | No change | 35% | 55% | 55% | 0.4417 | 0.5706 | 509 | 477 chars | 0% | 31.5% | 23,979 ms | 0.33 ms |
| 1 — Parser | pymupdf + join pages + strip noise | **40%** | **70%** | **75%** | **0.5350** | 0.5586 | 493 | 484 chars | 0% | 31.2% | 23,414 ms | 0.29 ms |
| 2 — Chunking | character size=1000 (metric winner) / section_wise size=1000 (production pick) | 65% | 80% | **90%** | **0.7475** | — | 16 ⚠️ | 12,866 chars | 68.8% | 83.7% | 1,392ms | 0.04ms |
| 3 — Embedding | _TBD_ | | | | | | | | | | | |
| 4 — Retrieval | _TBD_ | | | | | | | | | | | |
| 5 — Query | _TBD_ | | | | | | | | | | | |
| 6 — Hybrid Search | _TBD_ | | | | | | | | | | |

---

## Phase 0 — Baseline

**Branch:** `phase/0-baseline` | **Tag:** `phase-0-baseline`

**Configuration:** Recursive chunker · chunk_size=800 · overlap=80 · FAISS · pypdf loader · page-by-page parsing

### Results

| Metric | Value |
|--------|-------|
| Recall@1 | 35.00% |
| Recall@3 | 55.00% |
| Recall@5 | 55.00% |
| MRR | 0.4417 |
| Avg Top-1 Score | 0.5706 |
| Total chunks | 509 |
| Avg chunk size | 477 chars / 120 tokens |
| Std dev | 324 chars |
| Min / Max | 63 / 880 chars |
| Oversized (>384 tokens) | 0 (0.0%) |
| Token utilization | 31.5% |
| Parse time | 4,589 ms |
| Embed time | 23,979 ms |
| Index time | 2.9 ms |
| Avg query latency | 0.33 ms |

### Findings

- **Recall@5 = 55%** means 9 of 20 questions find the right answer in the top 5. Almost half are being missed entirely.
- **Recall@1 = 35%** — the correct chunk is ranked first only 7 times out of 20. MRR of 0.44 confirms the answer often lands at rank 2–3.
- **Token utilization is only 31.5%** — chunks average just 120 tokens against a 384-token model window. This signals under-packing, not over-packing. Larger chunk sizes may actually help.
- **High std dev (324 chars)** — chunk sizes vary wildly from 63 to 880 chars. Some chunks are too small to carry meaningful context; others approach the size limit.
- **Query latency is negligible (0.33 ms)** — FAISS brute-force is fast at this scale (~500 vectors).
- **No oversized chunks** — pypdf + recursive splitting keeps all chunks within the token limit, but at the cost of very small average sizes.

### Root Causes Identified

1. **Cross-page splitting** — pypdf returns one dict per page; the recursive chunker processes each page independently. Explanations that span a page boundary (e.g., multi-head attention in the Attention paper) get split across two incomplete chunks.
2. **Citation noise** — inline citations like `[13]`, `[35, 2, 5]` appear throughout every chunk and dilute embedding quality.
3. **Figure caption bleed-in** — captions like `"Figure 2: Scaled Dot-Product Attention"` get included in chunks without contributing meaningful semantic content.
4. **Query–document language gap** — questions are interrogative (`"What is multi-head attention?"`); the paper writes in formal declarative prose. The embedding model bridges this partially but not fully.

### Conclusion

The baseline pipeline is functional but leaves significant room for improvement. The most addressable issues are in the **parser layer** (cross-page splits, noise) and **chunking layer** (size calibration). The query–document language gap is the hardest to fix and will require Phase 5 (HyDE/RAG Fusion).

---

## Phase 1 — Parser

**Branch:** `phase/1-parser` | **Tag:** `phase-1-parser`

**What changed:** `shared/loader.py` only — switched from `pypdf` to `pymupdf`, joined all pages per PDF into one document before chunking, stripped figure captions and inline citations.

### Results

| Metric | Phase 0 | Phase 1 | Delta |
|--------|---------|---------|-------|
| Recall@1 | 35.00% | **40.00%** | +5% |
| Recall@3 | 55.00% | **70.00%** | +15% |
| Recall@5 | 55.00% | **75.00%** | +20% |
| MRR | 0.4417 | **0.5350** | +0.09 |
| Avg Top-1 Score | 0.5706 | 0.5586 | -0.01 |
| Total chunks | 509 | 493 | -16 |
| Avg chunk size | 477 chars | 484 chars | +7 |
| Oversized | 0% | 0% | — |
| Token utilization | 31.5% | 31.2% | -0.3% |
| Parse time | 4,589 ms | **383 ms** | -4,206 ms |
| Embed time | 23,979 ms | 23,414 ms | -565 ms |
| Avg query latency | 0.33 ms | 0.29 ms | -0.04 ms |

### Findings

- **Recall@5 jumped from 55% → 75%** — the single biggest gain so far. 4 additional questions now find the correct answer in the top 5.
- **Recall@3 improved the most (+15%)** — answers that were buried at rank 4–5 are now surfacing at rank 1–3. This is the clearest signal that joining pages fixed cross-boundary splits.
- **MRR improved from 0.44 → 0.54** — the correct chunk is now ranking closer to position 1 on average.
- **Avg Top-1 Score slightly dropped (0.5706 → 0.5586)** — cosmetically surprising, but explained by the fact that some previously easy questions now face stiffer competition from denser, cleaner chunks. The distribution of scores improved overall (better Recall@3/5), even though the single-question top-1 average dipped slightly.
- **Parse time dropped 12× (4,589 ms → 383 ms)** — pymupdf is significantly faster than pypdf for text extraction.
- **Chunk count barely changed (509 → 493)** — joining pages didn't drastically alter the number of chunks. The recursive splitter still produces similar-sized pieces; it just no longer stops at page boundaries.

### What drove the gain

The dominant factor was **joining pages per PDF**. The multi-head attention explanation in the Attention paper spans pages 3–4 — pypdf split it into two incomplete chunks, neither of which matched the query well. After joining, the recursive splitter can see across the boundary and keeps the explanation intact.

Citation stripping (`[13]`, `[35, 2, 5]`) and caption removal contributed modest gains by reducing embedding noise in every chunk.

### Conclusion

Parser quality has a large, low-effort impact on retrieval. Switching to a better extractor (pymupdf) and eliminating artificial page boundaries is one of the highest-ROI changes in the entire pipeline. **All subsequent phases will build on this improved parser.**

---

## Phase 2 — Chunking

**Branch:** `phase/2-chunking` | **Tag:** `phase-2-chunking`

**What changed:** Swept 3 chunkers (recursive, character, section_wise) × 4 chunk sizes (400, 800, 1000, 1200). Semantic excluded — too slow, not competitive. Overlap = 10% of chunk size. Parser from Phase 1 fixed throughout.

### Full Sweep Results

| Chunker | Size | Overlap | R@1 | R@3 | R@5 | MRR | Chunks | Avg Size | Oversized | Token Util | Embed Time |
|---------|------|---------|-----|-----|-----|-----|--------|----------|-----------|------------|------------|
| recursive | 400 | 40 | 20% | 40% | 55% | 0.3242 | 937 | 255 chars | 0.0% | 16.7% | 18,086ms |
| recursive | 800 | 80 | 40% | 70% | 75% | 0.5350 | 493 | 484 chars | 0.0% | 31.2% | 17,014ms |
| recursive | 1000 | 100 | 20% | 45% | 65% | 0.3558 | 399 | 596 chars | 0.3% | 38.1% | 20,963ms |
| recursive | 1200 | 120 | 25% | 55% | 80% | 0.4408 | 332 | 709 chars | 3.3% | 45.2% | 17,440ms |
| character | 400 | 40 | 70% | 80% | 85% | 0.7542 | 17 ⚠️ | 12,082 chars | 64.7% | 78.7% | 1,409ms |
| character | 800 | 80 | 60% | 75% | 85% | 0.7000 | 17 ⚠️ | 12,103 chars | 64.7% | 79.0% | 1,380ms |
| **character** | **1000** | **100** | **65%** | **80%** | **90%** | **0.7475** | **16 ⚠️** | **12,866 chars** | **68.8%** | **83.7%** | **1,392ms** |
| character | 1200 | 120 | 65% | 80% | 85% | 0.7292 | 16 ⚠️ | 12,877 chars | 68.8% | 83.8% | 1,245ms |
| section_wise | 400 | 40 | 45% | 55% | 75% | 0.5308 | 531 | 421 chars | 0.2% | 27.2% | 19,134ms |
| section_wise | 800 | 80 | 50% | 75% | 80% | 0.6183 | 273 | 818 chars | 0.4% | 52.4% | 17,816ms |
| section_wise | 1000 | 100 | 55% | 70% | 85% | 0.6492 | 214 | 1,039 chars | 2.3% | 65.6% | 17,201ms |
| section_wise | 1200 | 120 | 65% | 70% | 75% | 0.6875 | 174 | 1,271 chars | 13.8% | 78.1% | 16,653ms |

⚠️ = chunk_size parameter is ignored — actual chunks are ~12× larger than specified.

### Delta from Phase 1

| Metric | Phase 1 | Phase 2 | Delta |
|--------|---------|---------|-------|
| Recall@5 | 75% | **90%** | +15% |
| MRR | 0.5350 | **0.7475** | +0.21 |

### The Critical Finding — Character Chunker is Broken (in a useful way)

Character produces only **16 chunks** for the entire 4-paper corpus regardless of `chunk_size`. The reason: character splits exclusively on `\n\n`, and in these PDFs the sections between double newlines are enormous (avg 12,866 chars ≈ 3,200 tokens). The `chunk_size` parameter is completely ignored.

This means 68.8% of character chunks silently exceed the 384-token embedding window. Only the first ~1,500 chars of each chunk are actually embedded — the rest is invisible to retrieval. Yet R@5=90% because with only 16 chunks, the answer almost always appears in the first 1,500 chars of *some* section.

**This is coarse retrieval that happens to work on a small corpus — not a well-calibrated chunking strategy.**

### The Production-Safe Winner — Section-wise size=1000

Section_wise at size=1000 gives R@5=85%, MRR=0.6492 with 214 proper chunks, only 2.3% oversized, and sensible section-level granularity. It actually respects `chunk_size` and will scale predictably to larger corpora.

### Other Findings

- **Recursive is the most sensitive** — R@5 swings 55%→80% across sizes. It falls apart at size=400 (too fragmented) and size=1000 (paragraph–boundary misalignment).
- **Section_wise peaks at size=1000**, then degrades at 1200 (13.8% oversized — embedding truncation starts hurting).
- **Character embed time is 13× faster** (1,392ms vs ~17,000ms) — only 16 chunks to embed. Speed advantage is an artifact of under-chunking.

### Conclusion

**Metric winner:** character size=1000 → R@5=90%, MRR=0.7475 (but effectively doing document-level retrieval with heavy truncation).

**Production pick for remaining phases:** section_wise size=1000 → R@5=85%, MRR=0.6492 (proper chunking, 2.3% oversized, stable at scale).

All subsequent phases will use **section_wise size=1000 overlap=100** as the fixed chunker.

---

## Phase 3 — Embedding

**Branch:** `phase/3-embedding` | **Tag:** `phase-3-embedding`

**What changes:** `shared/embedder.py` only — compare embedding models:
- `all-mpnet-base-v2` (baseline, 768-dim, 384 tokens)
- `all-MiniLM-L6-v2` (faster, 384-dim, 256 tokens)
- `multi-qa-mpnet-base-dot-v1` (optimised for Q&A retrieval, 768-dim)

_Results to be recorded after model comparison._

---

## Phase 4 — Retrieval + Reranking

**Branch:** `phase/4-retrieval` | **Tag:** `phase-4-retrieval`

**What changes:** Add a cross-encoder reranker after initial bi-encoder retrieval.
- Stage 1: retrieve top-20 candidates with bi-encoder (fast, approximate)
- Stage 2: rerank all 20 with `cross-encoder/ms-marco-MiniLM-L-6-v2` (slower, precise)
- Return top-5 from reranked list

_Results to be recorded after reranker integration._

---

## Phase 5 — Query Improvement

**Branch:** `phase/5-query` | **Tag:** `phase-5-query`

**What changes:** Query preprocessing before embedding.
- **HyDE** (Hypothetical Document Embeddings) — LLM generates a fake answer, that answer is embedded instead of the raw question. Closes the question↔document language gap.
- **RAG Fusion** — generate 3–5 query variants, retrieve for each, merge results with Reciprocal Rank Fusion.

_Requires `OPENAI_API_KEY`. Results to be recorded after implementation._

---

## Phase 6 — Hybrid Search

**Branch:** `phase/6-hybrid` | **Tag:** `phase-6-hybrid`

**What changes:** Combine BM25 keyword search with dense vector search.
- BM25 excels at exact term matching (`"multi-head attention"` → exact hit)
- Dense search excels at semantic similarity
- Merge both ranked lists using Reciprocal Rank Fusion (RRF)

_Results to be recorded after BM25 + RRF integration._

---

## Key Takeaways

_To be filled after all phases are complete._

- Which phase produced the biggest single gain?
- Which changes had surprisingly little impact?
- What is the total improvement from Phase 0 → Phase 6?
- What would you prioritise in a production system?
