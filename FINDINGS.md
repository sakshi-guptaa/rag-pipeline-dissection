# RAG Retrieval Audit — Findings

Benchmark: 20 golden Q&A pairs across 4 academic papers.
Query fixed per phase. Only one pipeline layer changes per phase.
Baseline: Recursive chunker | chunk_size=800 | overlap=80 | FAISS

## Scorecard

| Phase | Note | Chunker | Size | Overlap | Store | R@1 | R@3 | R@5 | MRR | Top-1 | Chunks | Avg Size | Oversized | Token Util | Embed Time | Query Latency |
|-------|------|---------|------|---------|-------|-----|-----|-----|-----|-------|--------|----------|-----------|------------|------------|---------------|
| 0-Baseline | Default settings, no optimisation | recursive | 800 | 80 | faiss | 35.00% | 55.00% | 55.00% | 0.4417 | 0.5706 | 509 | 477 | 0.0% | 31.5% | 23980ms | 0.33ms |
| 1-Parser | pymupdf + join pages + strip captions & citations | recursive | 800 | 80 | faiss | 40.00% | 70.00% | 75.00% | 0.5350 | 0.5586 | 493 | 484 | 0.0% | 31.2% | 23414ms | 0.29ms |
| 2-Chunking | character size=1000 (winner of 23-combo sweep) | character | 1000 | 100 | faiss | — | — | **90.00%** | **0.7475** | — | — | — | — | — | — | — |

## Phase 2 — Chunking Sweep (23 combinations)

> Semantic size=1200 was excluded (diminishing returns observed, excessive runtime).

| Chunker | Size | Overlap | R@1 | R@3 | R@5 | MRR |
|---------|------|---------|-----|-----|-----|-----|
| recursive | 200 | 20 | — | — | 45% | 0.2267 |
| recursive | 400 | 40 | — | — | 55% | 0.3242 |
| recursive | 600 | 60 | — | — | 65% | 0.3992 |
| recursive | 800 | 80 | — | — | 75% | 0.5350 |
| recursive | 1000 | 100 | — | — | 65% | 0.3558 |
| recursive | 1200 | 120 | — | — | 80% | 0.4408 |
| character | 200 | 20 | — | — | 85% | 0.7292 |
| character | 400 | 40 | — | — | 85% | 0.7542 |
| character | 600 | 60 | — | — | 85% | 0.7225 |
| character | 800 | 80 | — | — | 85% | 0.7000 |
| **character** | **1000** | **100** | — | — | **90%** | **0.7475** |
| character | 1200 | 120 | — | — | 85% | 0.7292 |
| section_wise | 200 | 20 | — | — | 55% | 0.4125 |
| section_wise | 400 | 40 | — | — | 75% | 0.5308 |
| section_wise | 600 | 60 | — | — | 70% | 0.5792 |
| section_wise | 800 | 80 | — | — | 80% | 0.6183 |
| section_wise | 1000 | 100 | — | — | 85% | 0.6492 |
| section_wise | 1200 | 120 | — | — | 75% | 0.6875 |
| semantic | 200 | 20 | — | — | 60% | 0.3083 |
| semantic | 400 | 40 | — | — | 70% | 0.5792 |
| semantic | 600 | 60 | — | — | 80% | 0.5350 |
| semantic | 800 | 80 | — | — | 65% | 0.5500 |
| semantic | 1000 | 100 | — | — | 75% | 0.6000 |

**Winner: character · size=1000 · overlap=100 → Recall@5=90%, MRR=0.7475**
