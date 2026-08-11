"""TensorFlow two-tower recommendation model and helper functions."""
from __future__ import annotations
import numpy as np
import pandas as pd

def require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required for the two-tower experiment. Install requirements.txt.") from exc
    return tf

def build_vocabs(users: pd.DataFrame, assets: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "user_id": sorted(users.user_id.astype(str).unique()),
        "risk_profile": sorted(users.risk_profile.astype(str).unique()),
        "preferred_sector": sorted(users.preferred_sector.astype(str).unique()),
        "ticker": sorted(assets.ticker.astype(str).unique()),
        "asset_sector": sorted(assets.sector.astype(str).unique()),
    }

def build_two_tower_model(users: pd.DataFrame, assets: pd.DataFrame, embedding_dim: int = 32):
    tf = require_tensorflow()
    vocabs = build_vocabs(users, assets)
    def lookup(vocab):
        return tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None)
    user_id_in = tf.keras.Input(shape=(), dtype=tf.string, name="user_id")
    risk_in = tf.keras.Input(shape=(), dtype=tf.string, name="risk_profile")
    pref_in = tf.keras.Input(shape=(), dtype=tf.string, name="preferred_sector")
    age_in = tf.keras.Input(shape=(1,), dtype=tf.float32, name="age")
    exp_in = tf.keras.Input(shape=(1,), dtype=tf.float32, name="experience_years")
    def embed(inp, vocab, dim=embedding_dim):
        ids = lookup(vocab)(inp); return tf.keras.layers.Embedding(len(vocab)+1, dim)(ids)
    u = tf.keras.layers.Concatenate()([embed(user_id_in, vocabs["user_id"]), embed(risk_in, vocabs["risk_profile"], 8), embed(pref_in, vocabs["preferred_sector"], 8), age_in/100.0, exp_in/30.0])
    u = tf.keras.layers.Dense(64, activation="relu")(u); u = tf.keras.layers.Dense(embedding_dim)(u); u = tf.math.l2_normalize(u, axis=1)
    user_model = tf.keras.Model([user_id_in, risk_in, pref_in, age_in, exp_in], u, name="user_tower")

    ticker_in = tf.keras.Input(shape=(), dtype=tf.string, name="ticker")
    sector_in = tf.keras.Input(shape=(), dtype=tf.string, name="asset_sector")
    numeric_in = tf.keras.Input(shape=(6,), dtype=tf.float32, name="asset_numeric")
    a = tf.keras.layers.Concatenate()([embed(ticker_in, vocabs["ticker"]), embed(sector_in, vocabs["asset_sector"], 8), numeric_in])
    a = tf.keras.layers.Dense(64, activation="relu")(a); a = tf.keras.layers.Dense(embedding_dim)(a); a = tf.math.l2_normalize(a, axis=1)
    asset_model = tf.keras.Model([ticker_in, sector_in, numeric_in], a, name="asset_tower")

    inputs = {"user_id": user_id_in, "risk_profile": risk_in, "preferred_sector": pref_in, "age": age_in, "experience_years": exp_in, "ticker": ticker_in, "asset_sector": sector_in, "asset_numeric": numeric_in}
    score = tf.keras.layers.Dot(axes=1, normalize=False)([user_model([user_id_in, risk_in, pref_in, age_in, exp_in]), asset_model([ticker_in, sector_in, numeric_in])])
    probability = tf.keras.layers.Activation("sigmoid", name="interaction_probability")(score)
    model = tf.keras.Model(inputs, probability, name="financial_two_tower")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    return model, user_model, asset_model

def make_training_pairs(train: pd.DataFrame, users: pd.DataFrame, assets: pd.DataFrame, negatives_per_positive: int = 2, seed: int = 42):
    rng = np.random.default_rng(seed)
    user_map = users.set_index("user_id")
    asset_map = assets.set_index("ticker")
    numeric_cols = ["annual_return", "volatility", "beta", "pe_ratio", "dividend_yield", "market_cap_billion"]
    means = assets[numeric_cols].mean(); stds = assets[numeric_cols].std().replace(0, 1)
    all_tickers = assets.ticker.to_numpy(); rows=[]
    seen = train.groupby("user_id").ticker.apply(set).to_dict()
    for r in train.itertuples():
        rows.append((r.user_id, r.ticker, 1.0))
        candidates = [t for t in all_tickers if t not in seen[r.user_id]]
        for t in rng.choice(candidates, size=min(negatives_per_positive, len(candidates)), replace=False): rows.append((r.user_id, t, 0.0))
    features={k:[] for k in ["user_id","risk_profile","preferred_sector","age","experience_years","ticker","asset_sector","asset_numeric"]}; labels=[]
    for uid,ticker,label in rows:
        u=user_map.loc[uid]; a=asset_map.loc[ticker]
        features["user_id"].append(uid); features["risk_profile"].append(u.risk_profile); features["preferred_sector"].append(u.preferred_sector)
        features["age"].append([float(u.age)]); features["experience_years"].append([float(u.experience_years)])
        features["ticker"].append(ticker); features["asset_sector"].append(a.sector); features["asset_numeric"].append(((a[numeric_cols]-means)/stds).astype(float).to_list()); labels.append(label)
    return {k:np.array(v) for k,v in features.items()}, np.array(labels, dtype=np.float32), means, stds

def make_recommend_fn(model, users, assets, train, means, stds):
    numeric_cols=["annual_return","volatility","beta","pe_ratio","dividend_yield","market_cap_billion"]
    user_map=users.set_index("user_id")
    def recommend(user_id: str, k: int=10):
        u=user_map.loc[user_id]; candidates=assets[~assets.ticker.isin(set(train.loc[train.user_id==user_id,"ticker"]))].copy(); n=len(candidates)
        x={"user_id":np.repeat(user_id,n),"risk_profile":np.repeat(u.risk_profile,n),"preferred_sector":np.repeat(u.preferred_sector,n),"age":np.full((n,1),float(u.age)),"experience_years":np.full((n,1),float(u.experience_years)),"ticker":candidates.ticker.to_numpy(),"asset_sector":candidates.sector.to_numpy(),"asset_numeric":((candidates[numeric_cols]-means)/stds).to_numpy(dtype=np.float32)}
        scores=model.predict(x,verbose=0).reshape(-1); return candidates.iloc[np.argsort(-scores)].ticker.head(k).tolist()
    return recommend
