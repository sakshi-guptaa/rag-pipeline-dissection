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
