# RAG Retrieval Audit — Findings

Benchmark: 20 golden Q&A pairs across 4 academic papers.
Query fixed per phase. Only one pipeline layer changes per phase.
Baseline: Recursive chunker | chunk_size=800 | overlap=80 | FAISS

## Scorecard

| Phase | Note | Chunker | Size | Overlap | Store | R@1 | R@3 | R@5 | MRR | Top-1 | Chunks | Avg Size | Oversized | Token Util | Embed Time | Query Latency |
|-------|------|---------|------|---------|-------|-----|-----|-----|-----|-------|--------|----------|-----------|------------|------------|---------------|
| 0-Baseline | Default settings, no optimisation | recursive | 800 | 80 | faiss | 35.00% | 55.00% | 55.00% | 0.4417 | 0.5706 | 509 | 477 | 0.0% | 31.5% | 23980ms | 0.33ms |
