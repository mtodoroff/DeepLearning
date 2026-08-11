from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import RAW_DATA_DIR, RANDOM_SEED

SECTORS = ["Technology", "Financials", "Healthcare", "Energy", "Industrials", "Consumer", "Utilities", "Real Estate"]
RISK_LEVELS = ["conservative", "balanced", "aggressive"]

def generate_assets(n_assets: int = 60, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sectors = rng.choice(SECTORS, n_assets)
    volatility = np.clip(rng.normal(0.24, 0.09, n_assets), 0.06, 0.60)
    annual_return = np.clip(0.04 + 0.35 * volatility + rng.normal(0, 0.06, n_assets), -0.15, 0.45)
    beta = np.clip(0.55 + 2.1 * volatility + rng.normal(0, 0.15, n_assets), 0.35, 2.2)
    dividend = np.clip(0.055 - 0.11 * volatility + rng.normal(0, 0.009, n_assets), 0, 0.075)
    pe_ratio = np.clip(rng.lognormal(3.0, 0.45, n_assets), 5, 70)
    market_cap = np.exp(rng.normal(np.log(45), 1.15, n_assets)).clip(1, 2500)
    return pd.DataFrame({
        "ticker": [f"FAS{i:03d}" for i in range(1, n_assets + 1)],
        "company_name": [f"Synthetic Capital {i:03d}" for i in range(1, n_assets + 1)],
        "sector": sectors,
        "annual_return": annual_return.round(4),
        "volatility": volatility.round(4),
        "beta": beta.round(3),
        "pe_ratio": pe_ratio.round(2),
        "dividend_yield": dividend.round(4),
        "market_cap_billion": market_cap.round(2),
    })

def generate_users(n_users: int = 600, seed: int = RANDOM_SEED + 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    risk = rng.choice(RISK_LEVELS, n_users, p=[0.30, 0.45, 0.25])
    horizon_map = {"conservative": ["long", "medium"], "balanced": ["medium", "long"], "aggressive": ["short", "medium"]}
    horizons = [rng.choice(horizon_map[r]) for r in risk]
    exp = np.array([rng.integers(0, 8) if r == "conservative" else rng.integers(1, 16) for r in risk])
    return pd.DataFrame({
        "user_id": [f"U{i:05d}" for i in range(1, n_users + 1)],
        "age": rng.integers(21, 71, n_users),
        "experience_years": exp,
        "risk_profile": risk,
        "preferred_sector": rng.choice(SECTORS, n_users),
        "investment_horizon": horizons,
    })

def _asset_utility(user: pd.Series, assets: pd.DataFrame) -> np.ndarray:
    sector_match = (assets["sector"].to_numpy() == user["preferred_sector"]) * 1.4
    if user["risk_profile"] == "conservative":
        utility = -3.2 * assets["volatility"].to_numpy() + 8.0 * assets["dividend_yield"].to_numpy() - 0.25 * assets["beta"].to_numpy()
    elif user["risk_profile"] == "aggressive":
        utility = 4.2 * assets["annual_return"].to_numpy() + 1.7 * assets["volatility"].to_numpy() + 0.25 * assets["beta"].to_numpy()
    else:
        utility = 2.8 * assets["annual_return"].to_numpy() - 1.3 * assets["volatility"].to_numpy() + 3.0 * assets["dividend_yield"].to_numpy()
    return utility + sector_match + 0.08 * np.log1p(assets["market_cap_billion"].to_numpy())

def generate_interactions(users: pd.DataFrame, assets: pd.DataFrame, min_per_user: int = 8, max_per_user: int = 18, seed: int = RANDOM_SEED + 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = np.datetime64("2022-01-01")
    total_days = 365 * 3
    for _, user in users.iterrows():
        utility = _asset_utility(user, assets)
        utility = utility + rng.normal(0, 0.35, len(assets))
        probs = np.exp(utility - utility.max()); probs /= probs.sum()
        n = int(rng.integers(min_per_user, max_per_user + 1))
        chosen = rng.choice(len(assets), size=n, replace=False, p=probs)
        days = np.sort(rng.integers(0, total_days, size=n))
        for idx, day in zip(chosen, days):
            u = utility[idx]
            event = rng.choice(["view", "watchlist", "buy"], p=[0.43, 0.34, 0.23])
            weight = {"view": 1.0, "watchlist": 2.0, "buy": 4.0}[event]
            rows.append({
                "user_id": user["user_id"],
                "ticker": assets.iloc[idx]["ticker"],
                "event_type": event,
                "interaction_weight": weight,
                "timestamp": str(start + np.timedelta64(int(day), "D")),
            })
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

def generate_all(output_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = generate_assets(); users = generate_users(); interactions = generate_interactions(users, assets)
    assets.to_csv(output_dir / "assets.csv", index=False)
    users.to_csv(output_dir / "users.csv", index=False)
    interactions.to_csv(output_dir / "interactions.csv", index=False)
    return assets, users, interactions

if __name__ == "__main__":
    a, u, i = generate_all()
    print(f"Generated {len(a)} assets, {len(u)} users and {len(i)} interactions.")
