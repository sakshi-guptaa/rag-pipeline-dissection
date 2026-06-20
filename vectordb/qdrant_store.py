"""
Qdrant Vector Store

Real vector database: persistence, metadata filtering, payload indexing.
Uses in-memory mode by default (no Docker needed for demo).
Pass url="http://localhost:6333" for persistent Docker mode.

Indexing: HNSW (Hierarchical Navigable Small World) — approximate nearest
neighbor search using a multi-layer graph. Each vector is a node; edges
connect nearby vectors, with long-range shortcuts in upper layers for fast
traversal. Searches in O(log n). Trade-off: uses more memory than flat
indexes and recall is approximate (~99%+ with default settings, not 100%).
Distance metric: Cosine similarity.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)


class QdrantStore:
    def __init__(self, collection_name="chunks", dimension=384, url=None):
        if url:
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        self.dimension = dimension

        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

    def add(self, chunks, embeddings):
        points = []
        offset = self._count()
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            payload = {"text": chunk["text"]}
            payload.update(chunk.get("metadata", {}))
            points.append(PointStruct(
                id=offset + i,
                vector=emb.tolist() if hasattr(emb, "tolist") else emb,
                payload=payload,
            ))
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding, k=5, section_filter=None, source_filter=None):
        query = query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding

        conditions = []
        if section_filter:
            conditions.append(FieldCondition(key="section", match=MatchValue(value=section_filter)))
        if source_filter:
            conditions.append(FieldCondition(key="source", match=MatchValue(value=source_filter)))

        query_filter = Filter(must=conditions) if conditions else None

        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query,
            limit=k,
            query_filter=query_filter,
        )
        results = []
        for hit in hits.points:
            results.append({
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                "score": hit.score,
            })
        return results

    def _count(self):
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    @property
    def count(self):
        return self._count()
