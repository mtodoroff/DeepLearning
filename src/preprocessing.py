"""Validation, cleaning and temporal splitting utilities."""
from __future__ import annotations
import pandas as pd

REQUIRED_ASSET_COLUMNS = {"ticker", "sector", "annual_return", "volatility", "beta", "pe_ratio", "dividend_yield", "market_cap_billion"}
REQUIRED_USER_COLUMNS = {"user_id", "risk_profile", "preferred_sector"}
REQUIRED_INTERACTION_COLUMNS = {"user_id", "ticker", "event_type", "interaction_weight", "timestamp"}

def validate_frames(assets: pd.DataFrame, users: pd.DataFrame, interactions: pd.DataFrame) -> None:
    for name, frame, required in [("assets", assets, REQUIRED_ASSET_COLUMNS), ("users", users, REQUIRED_USER_COLUMNS), ("interactions", interactions, REQUIRED_INTERACTION_COLUMNS)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name} must not be empty")
    unknown_users = set(interactions.user_id) - set(users.user_id)
    unknown_assets = set(interactions.ticker) - set(assets.ticker)
    if unknown_users or unknown_assets:
        raise ValueError("Interactions contain unknown user or asset identifiers")

def clean_data(assets: pd.DataFrame, users: pd.DataFrame, interactions: pd.DataFrame):
    validate_frames(assets, users, interactions)
    assets = assets.drop_duplicates("ticker").copy()
    users = users.drop_duplicates("user_id").copy()
    interactions = interactions.drop_duplicates().copy()
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], errors="raise")
    interactions = interactions[interactions["interaction_weight"] > 0].sort_values("timestamp")
    return assets.reset_index(drop=True), users.reset_index(drop=True), interactions.reset_index(drop=True)

def temporal_user_split(interactions: pd.DataFrame, validation_items: int = 1, test_items: int = 1):
    if validation_items < 1 or test_items < 1:
        raise ValueError("validation_items and test_items must be positive")
    ordered = interactions.sort_values(["user_id", "timestamp"]).copy()
    rank_from_end = ordered.groupby("user_id").cumcount(ascending=False)
    test = ordered[rank_from_end < test_items]
    validation = ordered[(rank_from_end >= test_items) & (rank_from_end < test_items + validation_items)]
    train = ordered[rank_from_end >= test_items + validation_items]
    valid_users = train.groupby("user_id").size()
    keep = set(valid_users[valid_users > 0].index)
    return tuple(df[df.user_id.isin(keep)].reset_index(drop=True) for df in (train, validation, test))
