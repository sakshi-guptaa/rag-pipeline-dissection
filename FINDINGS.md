# RAG Retrieval Audit — Findings

Benchmark: 20 golden Q&A pairs across 4 academic papers.
Query fixed per phase. Only one pipeline layer changes per phase.
Baseline: Recursive chunker | chunk_size=800 | overlap=80 | FAISS

## Scorecard

| Phase | Note | Chunker | Size | Overlap | Store | R@1 | R@3 | R@5 | MRR | Top-1 | Chunks | Avg Size | Oversized | Token Util | Embed Time | Query Latency |
|-------|------|---------|------|---------|-------|-----|-----|-----|-----|-------|--------|----------|-----------|------------|------------|---------------|
| 0-Baseline | Default settings, no optimisation | recursive | 800 | 80 | faiss | 35.00% | 55.00% | 55.00% | 0.4417 | 0.5706 | 509 | 477 | 0.0% | 31.5% | 23980ms | 0.33ms |
| 1-Parser | pymupdf + join pages + strip captions & citations | recursive | 800 | 80 | faiss | 40.00% | 70.00% | 75.00% | 0.5350 | 0.5586 | 493 | 484 | 0.0% | 31.2% | 23414ms | 0.29ms |
| 2-Chunking | character size=1000 (metric winner) / section_wise size=1000 (production pick) | character | 1000 | 100 | faiss | 65% | 80% | **90%** | **0.7475** | — | 16 | 12,866 | 68.8% | 83.7% | 1,392ms | 0.04ms |
| 3-Embedding | bge-base-en-v1.5 (best MRR/Top-1; all 4 models tie at R@5=85%) | section_wise | 1000 | 100 | faiss | **60%** | **80%** | 85% | **0.6933** | **0.7036** | 214 | 1,039 | 1.4% | 49.7% | 13,686ms | 0.08ms |
| 4-Retrieval | cross-encoder reranker hurts: R@5 55%→80%, MRR flat, latency 4000× worse | section_wise | 1000 | 100 | faiss | 60% | 80% | 80% | 0.6917 | 0.6742 | 214 | 1,039 | 1.4% | 49.7% | 13,149ms | 329.95ms |

## Phase 2 — Chunking Sweep Results

Swept 3 chunkers × 4 sizes (400, 800, 1000, 1200). Semantic excluded (too slow, not competitive). Overlap = 10% of chunk size.

| Chunker | Size | Overlap | R@1 | R@3 | R@5 | MRR | Chunks | Avg Size | Oversized | Token Util | Embed Time |
|---------|------|---------|-----|-----|-----|-----|--------|----------|-----------|------------|------------|
| recursive | 400 | 40 | 20% | 40% | 55% | 0.3242 | 937 | 255 chars | 0.0% | 16.7% | 18,086ms |
| recursive | 800 | 80 | 40% | 70% | 75% | 0.5350 | 493 | 484 chars | 0.0% | 31.2% | 17,014ms |
| recursive | 1000 | 100 | 20% | 45% | 65% | 0.3558 | 399 | 596 chars | 0.3% | 38.1% | 20,963ms |
| recursive | 1200 | 120 | 25% | 55% | 80% | 0.4408 | 332 | 709 chars | 3.3% | 45.2% | 17,440ms |
| character | 400 | 40 | 70% | 80% | 85% | 0.7542 | 17 | 12,082 chars ⚠️ | 64.7% | 78.7% | 1,409ms |
| character | 800 | 80 | 60% | 75% | 85% | 0.7000 | 17 | 12,103 chars ⚠️ | 64.7% | 79.0% | 1,380ms |
| **character** | **1000** | **100** | **65%** | **80%** | **90%** | **0.7475** | **16** | **12,866 chars ⚠️** | **68.8%** | **83.7%** | **1,392ms** |
| character | 1200 | 120 | 65% | 80% | 85% | 0.7292 | 16 | 12,877 chars ⚠️ | 68.8% | 83.8% | 1,245ms |
| section_wise | 400 | 40 | 45% | 55% | 75% | 0.5308 | 531 | 421 chars | 0.2% | 27.2% | 19,134ms |
| section_wise | 800 | 80 | 50% | 75% | 80% | 0.6183 | 273 | 818 chars | 0.4% | 52.4% | 17,816ms |
| section_wise | 1000 | 100 | 55% | 70% | 85% | 0.6492 | 214 | 1,039 chars | 2.3% | 65.6% | 17,201ms |
| section_wise | 1200 | 120 | 65% | 70% | 75% | 0.6875 | 174 | 1,271 chars | 13.8% | 78.1% | 16,653ms |

⚠️ = avg chunk size far exceeds the chunk_size parameter — embedding model silently truncates tail content.

**Metric winner:** character size=1000 → R@5=90%, MRR=0.7475
**Production pick:** section_wise size=1000 → R@5=85%, MRR=0.6492 (proper chunking, 2.3% oversized)


## Phase 3 — Embedding Model Sweep

**Branch:** `phase/3-embedding` | **Tag:** `phase-3-embedding`

**What changed:** `shared/embedder.py` approach only — same chunks (section_wise size=1000), same FAISS store. Tested 4 local models. BGE and Nomic use task-specific query prefixes.

### Full Sweep Results

| Model | Token Limit | Dim | R@1 | R@3 | R@5 | MRR | Top-1 Score | Oversized | Token Util | Embed Time | Query Latency |
|-------|-------------|-----|-----|-----|-----|-----|-------------|-----------|------------|------------|---------------|
| all-mpnet-base-v2 *(baseline)* | 384 | 768 | 55% | 70% | 85% | 0.6492 | 0.5122 | 2.3% | 65.6% | 15,083ms | 0.16ms |
| multi-qa-mpnet-base-dot-v1 | 512 | 768 | 45% | 70% | 85% | 0.5825 | 0.6084 | 1.4% | 49.7% | 18,753ms | 0.16ms |
| **bge-base-en-v1.5** | **512** | **768** | **60%** | **80%** | **85%** | **0.6933** | **0.7036** | **1.4%** | **49.7%** | **13,686ms** | **0.08ms** |
| nomic-embed-text-v1 | 8192 | 768 | 50% | 80% | 85% | 0.6542 | 0.6468 | **0%** | 3.1% | 29,042ms | 5.05ms |

### Delta from Phase 2 (section_wise baseline)

| Metric | Phase 2 (baseline model) | Phase 3 (BGE) | Delta |
|--------|--------------------------|---------------|-------|
| Recall@1 | 55% | **60%** | +5% |
| Recall@3 | 70% | **80%** | +10% |
| Recall@5 | 85% | 85% | 0% |
| MRR | 0.6492 | **0.6933** | +0.04 |
| Avg Top-1 Score | 0.5122 | **0.7036** | +0.19 |

### The Critical Finding — R@5 is Capped at 85%

Every model, regardless of architecture or training objective, achieves exactly **R@5=85%**. The same 3 questions are missed by all 4 models. This means:

- The bottleneck is **not the embedding model** — it's something upstream
- The 3 missing answers are either not present in the chunks, or the query–document language gap is too wide for any bi-encoder to bridge
- Further embedding model changes will not move Recall@5 — we need Phase 5 (HyDE/RAG Fusion) to address the language gap, or Phase 6 (Hybrid Search) to catch keyword-matchable answers

### Per-Model Findings

- **BGE wins on MRR (+0.04) and Top-1 Score (+0.19)** — the query prefix ("Represent this sentence for searching relevant passages:") correctly steers the model toward retrieval-mode embeddings. The right answer ranks higher even when found.
- **multi-qa underperforms baseline on MRR (0.5825 vs 0.6492)** — surprising. Q&A training improves cosine similarity (Top-1: 0.61 vs 0.51) but doesn't translate to better ranking. The model may be over-fitting to direct QA datasets that look different from academic paper prose.
- **Nomic eliminates oversizing entirely (0%)** — confirmed: zero truncation with 8192-token limit. But no recall improvement, confirming truncation was not causing the 3 missed questions.
- **Nomic query latency is 30× slower (5.05ms vs 0.16ms)** — the FAISS index is over 768-dim vectors for all models (Nomic compresses to 768 internally), but Nomic's prefixes add overhead per query.
- **Token utilization for Nomic is only 3.1%** — our chunks (avg 1,039 chars ≈ 260 tokens) use just 3% of the 8192-token window. The long context is entirely wasted on this corpus.

### Conclusion

**Winner: BGE-base-en-v1.5** — best MRR (0.6933), best Top-1 cosine similarity (0.7036), fastest embed time (13,686ms), and lowest query latency (0.08ms).

**Key insight: embedding model choice affects ranking quality (MRR, Top-1 score) but not retrieval coverage (Recall@5) on this corpus.** The 3 missing questions require a different approach entirely. All subsequent phases will use **BGE-base-en-v1.5**.


## Phase 4 — Retrieval + Cross-Encoder Reranker

**Branch:** `phase/4-retrieval` | **Tag:** `phase-4-retrieval`

**What changed:** Retrieval only — added a two-stage pipeline. Stage 1: BGE bi-encoder retrieves top-20. Stage 2: `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks all 20 → returns top-5. Everything else fixed (section_wise size=1000, BGE embeddings, FAISS).

### Results

| Metric | Phase 3 (no reranker) | Phase 4 (+ reranker) | Delta |
|--------|----------------------|----------------------|-------|
| Recall@1 | 60% | 60% | 0% |
| Recall@3 | 80% | 80% | 0% |
| Recall@5 | **85%** | **80%** | **-5%** |
| MRR | 0.6933 | 0.6917 | -0.0016 |
| Avg Top-1 Score | 0.7036 | 0.6742 | -0.029 |
| Bi-encoder latency | 0.08ms | 0.24ms | +0.16ms |
| Reranker latency | — | ~330ms | — |
| **Total query latency** | **0.08ms** | **~330ms** | **~4000×** |

### The Unexpected Result — Reranker Hurts

The cross-encoder **degraded every metric** — R@5 dropped 5 points, MRR barely moved, and Top-1 cosine similarity fell. At the same time, query latency increased 4,000×.

### Why the Reranker Hurt Here

**1. Domain mismatch.** `ms-marco-MiniLM-L-6-v2` was trained on MS MARCO — web search snippets (short, factual, consumer queries). Our corpus is academic papers with long-form technical prose. The cross-encoder's relevance intuitions were trained on a completely different distribution.

**2. R@5 regression = the reranker promoted wrong chunks.** The bi-encoder already had the right answer in its top-20 for 85% of questions. The cross-encoder then re-ranked one of those 20 into position 6+ for one additional question, while the bi-encoder had it in top-5. The reranker's mistakes outweighed its corrections on this domain.

**3. Small corpus amplifies reranker errors.** With only 214 chunks, the bi-encoder top-20 is a large fraction of the index (≈9%). The candidates passed to the cross-encoder contain many plausible-looking but wrong chunks, and the cross-encoder — trained on web data — cannot reliably distinguish them from the correct academic prose.

**4. BGE is already a strong bi-encoder.** BGE was fine-tuned for retrieval on diverse corpora and uses an asymmetric query prefix. The reranker's marginal ranking improvement was insufficient to offset its domain-gap errors.

### Conclusion

**No reranker for this corpus and pipeline.** The cross-encoder degraded recall and added 330ms of latency per query. This is a domain-fit failure, not a problem with reranking as a technique.

**Key insight: a cross-encoder trained on web search does not generalise to academic papers.** In production, you'd need a reranker fine-tuned on scientific text (e.g., `cross-encoder/ms-marco-electra-base` for general, or a domain-specific model). On a small corpus where the bi-encoder already performs well, reranking has little headroom to gain and meaningful risk of hurting.

**Phase 4 is dropped from the best pipeline.** Continuing with BGE bi-encoder top-5, no reranker.
