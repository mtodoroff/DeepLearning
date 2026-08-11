"""Top-K recommender evaluation metrics."""
from __future__ import annotations
from collections.abc import Callable
import numpy as np
import pandas as pd

def evaluate_recommender(test: pd.DataFrame, recommend_fn: Callable[[str, int], list[str]], k: int = 10) -> dict[str, float]:
    precisions, recalls, reciprocal_ranks, hits = [], [], [], []
    recommended_catalog = set()
    for user_id, group in test.groupby("user_id"):
        relevant = set(group.ticker)
        recs = list(recommend_fn(user_id, k))[:k]
        recommended_catalog.update(recs)
        matched = [r for r in recs if r in relevant]
        precisions.append(len(matched) / k)
        recalls.append(len(matched) / len(relevant))
        hits.append(float(bool(matched)))
        rr = 0.0
        for rank, ticker in enumerate(recs, start=1):
            if ticker in relevant:
                rr = 1.0 / rank; break
        reciprocal_ranks.append(rr)
    return {
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
        f"hit_rate@{k}": float(np.mean(hits)),
        f"mrr@{k}": float(np.mean(reciprocal_ranks)),
        "unique_recommended_assets": float(len(recommended_catalog)),
    }
