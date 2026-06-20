# RAG Pipeline Dissection — Findings

Systematic, one-layer-at-a-time audit of what actually moves the needle in RAG retrieval quality.

**Corpus:** 4 academic papers (Attention Is All You Need, RAG, RAGAS, HNSW)  
**Eval:** 20 hand-crafted question–evidence pairs, fixed across all phases  
**Metrics:** Recall@1/3/5 (was the correct chunk retrieved?), MRR (how high did it rank?)

---

## Scorecard

| Phase | Change | R@1 | R@3 | R@5 | MRR |
|-------|--------|-----|-----|-----|-----|
| 0 — Baseline | pypdf · recursive chunker · all-mpnet | 35% | 55% | 55% | 0.44 |
| 1 — Parser | pymupdf · join pages · strip noise | 40% | 70% | 75% | 0.54 |
| 2 — Chunking | section_wise size=1000 overlap=100 | 55% | 70% | 85% | 0.65 |
| 3 — Embedding | BGE-base-en-v1.5 | 60% | 80% | 85% | 0.69 |
| 4 — Reranker | cross-encoder/ms-marco (dropped — hurts) | 60% | 80% | 80% | 0.69 |
| 5 — Hybrid | BM25 + dense weighted RRF (α=0.7) | 60% | 80% | **95%** | 0.71 |
| 6 — Query | HyDE (gpt-4o-mini) | **80%** | **85%** | **95%** | **0.83** |

> R@5 corrected from 90%→95% (Ph5) and 100% reached after fixing 2 evaluation bugs — see Addendum.

---

## Phase 0 — Baseline

**Setup:** pypdf (page-by-page) · recursive chunker (size=800, overlap=80) · all-mpnet-base-v2 · FAISS

**Result:** R@5=55%, MRR=0.44. Nearly half of questions miss the top-5 entirely.

**Root causes:** pypdf splits the document page-by-page, so explanations that span page boundaries become two incomplete chunks. Inline citations (`[13]`, `[35]`) and figure captions add noise to every embedding. Token utilization was only 31.5% — chunks averaged 120 tokens against a 384-token model window, signalling under-packing rather than over-packing (larger chunks could help).

---

## Phase 1 — Parser

**What:** Switched `shared/loader.py` from pypdf → pymupdf, joined all pages per PDF into one document before chunking, stripped figure captions and citation references.

**Why:** Cross-page splits were a structural problem no downstream fix could overcome. If the right text is split across two chunks, retrieval fails regardless of embedding model or chunk size.

**Result:**

| | Baseline | After | Δ |
|--|--|--|--|
| R@5 | 55% | **75%** | +20% |
| MRR | 0.44 | **0.54** | +0.10 |
| Parse time | 4,589ms | **383ms** | 12× faster |

**Takeaway:** Parser quality is the highest-ROI fix in the whole pipeline. Joining pages and removing noise is free and permanent. **All subsequent phases build on this parser.**

---

## Phase 2 — Chunking

**What:** Swept 3 chunkers (recursive, character, section_wise) × 4 sizes (400/800/1000/1200). Fixed overlap at 10% of chunk size.

**Why:** Chunk size affects both what fits in an embedding and how much context surrounds each answer.

**Key results (section_wise, the production pick):**

| Size | R@5 | MRR | Chunks | Oversized |
|------|-----|-----|--------|-----------|
| 400 | 75% | 0.53 | 531 | 0.2% |
| 800 | 80% | 0.62 | 273 | 0.4% |
| **1000** | **85%** | **0.65** | **214** | **2.3%** |
| 1200 | 75% | 0.69 | 174 | 13.8% |

**Hidden trap — character chunker:** The character chunker *appeared* to win (R@5=90%) but produced only 16 chunks averaging 12,866 chars — 68% of content silently truncated by the 384-token embedding limit. Good metrics on a small corpus, would collapse at scale.

**Takeaway:** section_wise size=1000 is the production pick — sensible granularity, 2.3% oversized, stable at scale. **All subsequent phases use this chunker.**

---

## Phase 3 — Embedding Model

**What:** Tested 4 local models on the same chunks. BGE and Nomic use task-specific query prefixes.

**Why:** Retrieval-optimised models should understand that queries and documents are asymmetric.

| Model | R@5 | MRR | Top-1 Sim |
|-------|-----|-----|-----------|
| all-mpnet-base-v2 (baseline) | 85% | 0.65 | 0.51 |
| multi-qa-mpnet | 85% | 0.58 | 0.61 |
| **bge-base-en-v1.5** | **85%** | **0.69** | **0.70** |
| nomic-embed-text-v1 | 85% | 0.65 | 0.65 |

**Critical finding:** Every model plateaued at exactly R@5=85%, missing the same 3 questions. The ceiling wasn't the embedding model — it was the retrieval *modality*: two questions needed exact keyword matching (→ BM25), one needed the query–document register gap closed (→ HyDE).

**Takeaway:** BGE wins on MRR and similarity scores (retrieval-tuned prefix matters). But R@5 won't move by swapping models — the problem is elsewhere. **All subsequent phases use BGE-base-en-v1.5.**

---

## Phase 4 — Cross-Encoder Reranker

**What:** Two-stage retrieval: BGE bi-encoder retrieves top-20, then `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks → top-5.

**Why:** Cross-encoders jointly encode query+document and can catch semantic nuances a bi-encoder misses.

| | Without reranker | With reranker |
|--|--|--|
| R@5 | 85% | **80%** |
| MRR | 0.69 | 0.69 |
| Query latency | 0.08ms | **330ms** |

**Why it hurt:** ms-marco was trained on web search snippets, not academic prose. Its relevance judgements don't transfer — it actively demoted correct chunks. 4,000× latency increase, negative recall impact.

**Takeaway:** Off-the-shelf rerankers are not plug-and-play. They need in-domain training. **Reranker dropped for all subsequent phases.**

---

## Phase 5 — Hybrid Search (BM25 + Dense + RRF)

**What:** Combined BM25 keyword retrieval with BGE dense retrieval using weighted Reciprocal Rank Fusion. Swept α (dense weight) from 0.5 to 0.8.

**Why:** Dense and BM25 have complementary failure modes — dense fails on exact-term questions, BM25 fails on paraphrases. RRF can capture both signals.

| α | R@1 | R@3 | R@5 | MRR |
|---|-----|-----|-----|-----|
| Dense only | 60% | 80% | 85% | 0.69 |
| BM25 only | 65% | 75% | 85% | 0.72 |
| **0.7** | **60%** | 80% | **95%** | **0.71** |
| 0.5 | 55% | **85%** | 95% | 0.71 |

> α=0.7 means 70% dense weight, 30% BM25 weight in the RRF score.

**Takeaway:** α=0.7 is the sweet spot — dense controls top slots (R@1 intact), BM25's 30% weight surfaces keyword-matchable answers. **First method to break the 85% ceiling.** BM25-only MRR (0.72) beating dense (0.69) shows exact terminology matching is powerful on academic text.

---

## Phase 6 — Query Expansion (HyDE + RAG Fusion)

**What:** At query time, used GPT-4o-mini to improve the query before retrieval. Tested 4 strategies on the same index (section_wise + BGE + hybrid α=0.7).

**Why:** The fundamental mismatch — questions are interrogative, documents are declarative prose — can't be fixed at the index. Fix it at query time instead.

| Strategy | R@1 | R@3 | R@5 | MRR | Latency |
|----------|-----|-----|-----|-----|---------|
| Hybrid (Ph5 baseline) | 60% | 80% | 95% | 0.71 | 34ms |
| **HyDE** | **80%** | **85%** | **95%** | **0.83** | 122ms |
| RAG Fusion (4 variants) | 50% | 80% | 85% | 0.65 | 86ms |
| HyDE + Fusion | 75% | 75% | 85% | 0.78 | 9,211ms |

**HyDE** generates a short fake academic paragraph answering the question, then embeds that instead of the raw query. The fake paragraph uses the same register as the document, eliminating the query–document language gap.

**RAG Fusion** generates 4 query paraphrases and merges the result lists via RRF. On this corpus it *hurt* — precise academic questions are already well-formed; paraphrases retrieved overlapping but noisier results that diluted the best signal.

**Takeaway:** HyDE alone: +20pp R@1, +0.12 MRR, 1 GPT call, 122ms. The biggest single-phase gain of the entire audit. Skip RAG Fusion on precise corpora.

**Non-determinism caveat:** GPT generates a different hypothetical answer each run (temperature=0.3), so MRR varies ~0.81–0.83 across runs. A single benchmark number for HyDE isn't reliable — run it 3× and report the range.

---

## Final Pipeline & Total Gain

```
PDF → pymupdf (join pages, strip noise)
    → section_wise chunker (size=1000, overlap=100)
    → BGE-base-en-v1.5 (query prefix for retrieval)
    → FAISS + BM25 hybrid (weighted RRF α=0.7)
    → HyDE query expansion (gpt-4o-mini, 1 call/query)
    → top-5 results
```

| Metric | Baseline | Final | Gain |
|--------|----------|-------|------|
| Recall@1 | 35% | **80%** | +45pp |
| Recall@3 | 55% | **85%** | +30pp |
| Recall@5 | 55% | **95%** | +40pp |
| MRR | 0.44 | **0.83** | +0.39 |
| Parse time | 4,589ms | **383ms** | 12× faster |

**What moved the needle most:**
1. **Parser** — biggest coverage jump (+20pp R@5). Free, permanent.
2. **HyDE** — biggest ranking jump (+20pp R@1, +0.12 MRR). 1 GPT call per query.
3. **Hybrid BM25** — broke the 85% ceiling that 4 embedding models couldn't crack.

**What didn't work:** cross-encoder reranker (domain mismatch, −5pp), RAG Fusion (noise on precise queries).

---

## Addendum: Evaluation Bug

After all phases, a diagnostic script found the 2 "still-failing" questions weren't retrieval failures — both correct chunks were being retrieved at rank 1–2. The eval's exact substring check was failing:

| Q | Wrong evidence string | Actual chunk text |
|---|-----------------------|-------------------|
| Q8: How does multi-head attention work? | `"different representation subspaces"` | `"representation\nsubspaces"` (newline from PDF) |
| Q20: What does RAGAS measure about context? | `"context relevancy"` | `"context relevance"` (synonym) |

Fixed both strings. Corrected Phase 5 hybrid R@5: 90% → **100%** (all 20 retrievable by Phase 5).

**Lesson for eval design:** Use verbatim quotes of ≥10 words as evidence strings, not short phrases. A single synonym or PDF-inserted newline silently makes correct retrieval look like failure.
