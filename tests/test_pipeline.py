import pandas as pd
from src.data_generation import generate_assets, generate_users, generate_interactions
from src.preprocessing import clean_data, temporal_user_split
from src.evaluation import evaluate_recommender
import pytest

def test_generated_data_and_split():
    assets=generate_assets(20); users=generate_users(30); interactions=generate_interactions(users,assets,6,8)
    assets,users,interactions=clean_data(assets,users,interactions)
    train,val,test=temporal_user_split(interactions)
    assert len(assets)==20 and len(users)==30
    assert not train.empty and not val.empty and not test.empty
    assert train.timestamp.max() <= test.timestamp.max()

def test_metrics_perfect_recommendation():
    test=pd.DataFrame({"user_id":["u1","u2"],"ticker":["a","b"]})
    rec=lambda uid,k: ["a"] if uid=="u1" else ["b"]
    result=evaluate_recommender(test,rec,k=1)
    assert result["hit_rate@1"]==1.0
    assert result["mrr@1"]==1.0

def test_temporal_split_has_no_user_level_leakage():
    assets = generate_assets(20)
    users = generate_users(30)
    interactions = generate_interactions(users, assets, 6, 8)

    _, _, interactions = clean_data(
        assets, users, interactions
    )

    train, val, test = temporal_user_split(interactions)

    check = pd.DataFrame({
        "train_max": train.groupby("user_id").timestamp.max(),
        "val_time": val.groupby("user_id").timestamp.min(),
        "test_time": test.groupby("user_id").timestamp.min(),
    })

    assert (check.train_max <= check.val_time).all()
    assert (check.val_time <= check.test_time).all()
    
def test_clean_data_removes_duplicates():
    assets = generate_assets(10)
    users = generate_users(10)
    interactions = generate_interactions(
        users, assets, 4, 6
    )

    duplicated = pd.concat(
        [interactions, interactions.iloc[[0]]],
        ignore_index=True
    )

    _, _, cleaned = clean_data(
        assets, users, duplicated
    )

    assert len(cleaned) == len(interactions)
    

def test_clean_data_rejects_unknown_user():
    assets = generate_assets(10)
    users = generate_users(10)
    interactions = generate_interactions(
        users, assets, 4, 6
    )

    interactions.loc[0, "user_id"] = "UNKNOWN_USER"

    with pytest.raises(ValueError):
        clean_data(assets, users, interactions)
