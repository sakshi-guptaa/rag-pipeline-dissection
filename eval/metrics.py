"""
Evaluation Metrics

recall@k: fraction of questions where the evidence phrase appears
          in at least one of the top-k retrieved chunks.

mrr:      mean reciprocal rank — average of 1/rank of the first
          chunk containing the evidence. 1.0 = always first.

latency:  query time in milliseconds.
"""


def recall_at_k(results, evidence, k=5):
    # "Did we find the answer somewhere in the top-k results?"
    # 1.0 = yes, the evidence appeared in at least one chunk. 0.0 = no, we missed it entirely.
    for r in results[:k]:
        if evidence.lower() in r["text"].lower():
            return 1.0
    return 0.0


def reciprocal_rank(results, evidence, k=5):
    # "How high did the right answer rank?"
    # 1.0 = it was the very first result. 0.5 = second. 0.33 = third. 0.0 = not in top-k at all.
    for i, r in enumerate(results[:k]):
        if evidence.lower() in r["text"].lower():
            return 1.0 / (i + 1)
    return 0.0


def evaluate_store(store, query_embedding_fn, golden, k=5):
    recalls = []
    mrrs = []
    for item in golden:
        qe = query_embedding_fn(item["question"])
        results = store.search(qe, k=k)
        recalls.append(recall_at_k(results, item["evidence"], k))
        mrrs.append(reciprocal_rank(results, item["evidence"], k))
    return {
        f"recall@{k}": sum(recalls) / len(recalls) if recalls else 0,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else 0,
    }
