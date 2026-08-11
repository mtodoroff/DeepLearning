"""Simple recommendation baselines."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def popularity_scores(train: pd.DataFrame) -> pd.Series:
    return train.groupby("ticker")["interaction_weight"].sum().sort_values(ascending=False)

def recommend_popular(train: pd.DataFrame, user_id: str, k: int = 10) -> list[str]:
    seen = set(train.loc[train.user_id == user_id, "ticker"])
    return [t for t in popularity_scores(train).index if t not in seen][:k]

def recommend_sector_popular(train: pd.DataFrame, users: pd.DataFrame, assets: pd.DataFrame, user_id: str, k: int = 10) -> list[str]:
    pref = users.set_index("user_id").loc[user_id, "preferred_sector"]
    tickers = set(assets.loc[assets.sector == pref, "ticker"])
    ranked = popularity_scores(train)
    seen = set(train.loc[train.user_id == user_id, "ticker"])
    first = [t for t in ranked.index if t in tickers and t not in seen]
    rest = [t for t in ranked.index if t not in seen and t not in first]
    return (first + rest)[:k]

def recommend_content(train: pd.DataFrame, assets: pd.DataFrame, user_id: str, k: int = 10) -> list[str]:
    feature_cols = ["annual_return", "volatility", "beta", "pe_ratio", "dividend_yield", "market_cap_billion"]
    x = StandardScaler().fit_transform(assets[feature_cols])
    asset_ids = assets.ticker.to_numpy()
    seen_rows = train[train.user_id == user_id]
    seen = set(seen_rows.ticker)
    if not seen:
        return list(asset_ids[:k])
    index = {t: i for i, t in enumerate(asset_ids)}
    weighted = [(x[index[t]], w) for t, w in zip(seen_rows.ticker, seen_rows.interaction_weight) if t in index]
    profile = np.average(np.vstack([v for v, _ in weighted]), axis=0, weights=[w for _, w in weighted])
    scores = x @ profile / ((np.linalg.norm(x, axis=1) + 1e-9) * (np.linalg.norm(profile) + 1e-9))
    order = np.argsort(-scores)
    return [asset_ids[i] for i in order if asset_ids[i] not in seen][:k]
