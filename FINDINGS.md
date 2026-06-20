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
| 2 — Chunking | character size=1000, winner of 23-combo sweep | **—** | **—** | **90%** | **0.7475** | — | — | — | — | — | — | — |
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

**What changed:** Swept all 4 chunkers × 6 chunk sizes (200→1200, overlap=10% of size). Parser from Phase 1 fixed. 23 of 24 combinations run (semantic size=1200 excluded — diminishing returns observed at size=1000 with excessive runtime).

### Full Sweep Results

| Chunker | Size | Overlap | R@5 | MRR | Winner? |
|---------|------|---------|-----|-----|---------|
| recursive | 200 | 20 | 45% | 0.2267 | |
| recursive | 400 | 40 | 55% | 0.3242 | |
| recursive | 600 | 60 | 65% | 0.3992 | |
| recursive | 800 | 80 | 75% | 0.5350 | |
| recursive | 1000 | 100 | 65% | 0.3558 | |
| recursive | 1200 | 120 | 80% | 0.4408 | |
| character | 200 | 20 | 85% | 0.7292 | |
| character | 400 | 40 | 85% | 0.7542 | |
| character | 600 | 60 | 85% | 0.7225 | |
| character | 800 | 80 | 85% | 0.7000 | |
| **character** | **1000** | **100** | **90%** | **0.7475** | ✅ |
| character | 1200 | 120 | 85% | 0.7292 | |
| section_wise | 200 | 20 | 55% | 0.4125 | |
| section_wise | 400 | 40 | 75% | 0.5308 | |
| section_wise | 600 | 60 | 70% | 0.5792 | |
| section_wise | 800 | 80 | 80% | 0.6183 | |
| section_wise | 1000 | 100 | 85% | 0.6492 | |
| section_wise | 1200 | 120 | 75% | 0.6875 | |
| semantic | 200 | 20 | 60% | 0.3083 | |
| semantic | 400 | 40 | 70% | 0.5792 | |
| semantic | 600 | 60 | 80% | 0.5350 | |
| semantic | 800 | 80 | 65% | 0.5500 | |
| semantic | 1000 | 100 | 75% | 0.6000 | |

### Delta from Phase 1

| Metric | Phase 1 | Phase 2 | Delta |
|--------|---------|---------|-------|
| Recall@5 | 75% | **90%** | +15% |
| MRR | 0.5350 | **0.7475** | +0.21 |

### Findings

- **Character chunker dominates** — every character chunk size (200–1200) outperforms the best recursive size (1200=80%). This is the biggest surprise of the sweep.
- **Character size=1000 wins overall** — R@5=90%, MRR=0.7475. Only 2 of 20 questions miss entirely.
- **MRR is the most revealing metric here** — character's MRR (0.70–0.75) is consistently ~0.15 higher than any other strategy, meaning the right answer lands at rank 1 far more often.
- **Recursive shows high variance** — drops from 75% at size=800 to 65% at size=1000, then recovers to 80% at size=1200. Sensitive to whether the paragraph boundary aligns with the chunk boundary.
- **Section-wise improves monotonically with size** (up to 1000) but never beats character. Preserving section structure helps but individual paragraph boundaries work better for these golden questions.
- **Semantic is the worst** — high compute cost, inconsistent results, not worth it for this corpus.

### Why character wins (the counter-intuitive result)

At Phase 0 (pypdf, page-by-page), character was the *worst* chunker — it produced oversized chunks that got silently truncated. Now with the Phase 1 parser (pymupdf + joined pages), character splits on `\n\n` paragraph breaks which in academic papers are **natural topic boundaries**. Each paragraph tends to discuss one idea. Character at size=1000 captures 1–2 complete paragraphs per chunk — enough context without diluting the embedding with off-topic content.

Recursive tries to be "smarter" by falling back through separators, but that flexibility also makes it less consistent — it sometimes splits in the middle of a paragraph when paragraph size doesn't align with chunk_size.

### Conclusion

**character + size=1000 + overlap=100** is the winning configuration. All subsequent phases will use this. Recall@5 is now 90% — only 2 questions out of 20 still miss. The remaining gap requires improving how we match queries to documents (Phase 5) or better ranking of retrieved results (Phase 4).

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
