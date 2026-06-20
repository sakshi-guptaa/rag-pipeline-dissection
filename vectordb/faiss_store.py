"""
FAISS Vector Store

Library-level nearest neighbor search. No server, no persistence,
no metadata filtering. Fast and simple — the baseline.

Indexing: IndexFlatIP (Flat Inner Product) — brute-force exact search.
Compares the query against every vector in the index. No approximation,
no graph structure, no quantization. O(n) per query. Guarantees perfect
recall but doesn't scale past ~1M vectors without switching to IVF or HNSW.
Vectors are L2-normalized before insert, so inner product = cosine similarity.
"""

import numpy as np
import faiss


class FaissStore:
    def __init__(self, dimension=384):
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []
        self.dimension = dimension

    def add(self, chunks, embeddings):
        embeddings = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding, k=5):
        query = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "text": self.chunks[idx]["text"],
                "metadata": self.chunks[idx]["metadata"],
                "score": float(score),
            })
        return results

    @property
    def count(self):
        return self.index.ntotal
