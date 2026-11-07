"""Tests for the member-B data pipeline. Uses the official 10k-row sample
(``uv run python scripts/prepare_data.py --sample``)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ope.data import (
    DataConfig,
    build_campaign,
    load_bandit_feedback,
    on_policy_value,
    policy_value_from_random_logs,
    uniform_action_dist,
    validate_action_dist,
)
from ope.data.features import OTHER, ContextEncoder
from ope.data.ground_truth import bootstrap_mean_ci, wilson_ci
from ope.data.io import align_item_context, clean_logs, read_item_context
from ope.data.dataset import time_split_bounds

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"
needs_sample = pytest.mark.skipif(not (SAMPLE / "random" / "men" / "men.csv").exists(), reason="sample not downloaded")


# ---------------------------------------------------------------- unit tests
def _toy_logs(n=50, campaign="men"):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2019-11-24", periods=n, freq="min", tz="UTC").astype(str)[::-1],
            "item_id": rng.integers(0, 34, n),
            "position": rng.integers(1, 4, n),
            "click": rng.integers(0, 2, n),
            "propensity_score": np.full(n, 1 / 34),
        }
    )
    for k in range(4):
        df[f"user_feature_{k}"] = rng.choice(["a", "b", "c"], n)
    for k in range(34):
        df[f"user-item_affinity_{k}"] = rng.integers(0, 3, n).astype("float32")
    return df


def test_clean_logs_drops_bad_rows_and_sorts():
    df = _toy_logs()
    df.loc[0, "item_id"] = 99          # out of range
    df.loc[1, "position"] = 4          # bad slot
    df.loc[2, "propensity_score"] = 0  # invalid pscore
    df.loc[3, "click"] = 2             # bad reward
    df = pd.concat([df, df.iloc[[10]]])  # duplicate
    out, rep = clean_logs(df, "men")
    assert rep["dropped_bad_item"] == 1 and rep["dropped_bad_position"] == 1
    assert rep["dropped_bad_pscore"] == 1 and rep["dropped_bad_click"] == 1
    assert rep["dropped_duplicate"] == 1 and len(out) == 51 - 1 - 4
    assert out["position"].between(0, 2).all()            # 0-indexed
    assert out["timestamp"].is_monotonic_increasing       # sorted by time


def test_time_split_is_chronological_and_complete():
    b = time_split_bounds(100, 0.1, 0.3)
    assert b == {"train": (0, 60), "val": (60, 70), "test": (70, 100)}
    assert "val" not in time_split_bounds(100, 0.0, 0.3)


def test_encoder_unknown_category_goes_to_other():
    df, _ = clean_logs(_toy_logs(), "men")
    enc = ContextEncoder(affinity_cols=[f"user-item_affinity_{k}" for k in range(34)], min_category_freq=1).fit(df)
    new = df.head(3).copy()
    new["user_feature_0"] = pd.Categorical(["zzz", "a", "zzz"])
    X = enc.transform(new)
    other_col = enc.vocab["user_feature_0"].index(OTHER)
    assert X[0, other_col] == 1 and X[2, other_col] == 1
    assert X.shape == (3, len(enc.feature_names))
    # one-hot block has exactly one 1 per user feature
    assert np.allclose(X[:, : enc.n_onehot].sum(1), 4)


def test_encoder_standardizes_on_fit_data():
    df, _ = clean_logs(_toy_logs(500), "men")
    enc = ContextEncoder(affinity_cols=[f"user-item_affinity_{k}" for k in range(34)]).fit(df)
    cont = enc.transform(df)[:, enc.n_onehot :]
    assert np.allclose(cont.mean(0), 0, atol=1e-4) and np.allclose(cont.std(0), 1, atol=1e-3)


def test_wilson_and_bootstrap_cover_true_rate():
    rng = np.random.default_rng(1)
    r = (rng.random(200_000) < 0.004).astype(int)
    lo, hi = wilson_ci(r.sum(), r.size)
    blo, bhi, boots = bootstrap_mean_ci(r, 500)
    assert lo < 0.004 < hi and blo < 0.004 < bhi
    # sparse bootstrap must match the analytic standard error
    assert abs(boots.std() - np.sqrt(r.mean() * (1 - r.mean()) / r.size)) / boots.std() < 0.15


def test_ips_ground_truth_recovers_true_value_on_synthetic_random_logs():
    """Simulate uniform logs with known click probs; GT of a deterministic
    policy must match its analytic value."""
    rng = np.random.default_rng(2)
    n, K, L = 300_000, 5, 3
    ctr = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    a = rng.integers(0, K, n)
    bf = {
        "n_rounds": n, "n_actions": K, "len_list": L,
        "action": a, "position": rng.integers(0, L, n),
        "reward": (rng.random(n) < ctr[a]).astype(int), "pscore": np.full(n, 1 / K),
    }
    best = np.zeros((n, K))
    best[:, 4] = 1.0  # always pick item 4 -> true value 0.05
    res = policy_value_from_random_logs(bf, best, n_bootstrap=200)
    assert res["ci_low"] < 0.05 < res["ci_high"]
    assert res["max_weight"] == K
    uni = policy_value_from_random_logs(bf, uniform_action_dist(n, K, L), n_bootstrap=10)
    assert np.isclose(uni["value"], bf["reward"].mean())  # uniform policy == on-policy CTR


def test_ground_truth_rejects_non_uniform_logs_and_bad_dists():
    bf = {"n_rounds": 2, "n_actions": 2, "len_list": 1, "action": np.array([0, 1]),
          "position": np.array([0, 0]), "reward": np.array([1, 0]), "pscore": np.array([0.9, 0.1])}
    with pytest.raises(ValueError, match="uniform"):
        policy_value_from_random_logs(bf, np.full((2, 2), 0.5))
    with pytest.raises(ValueError, match="sum to 1"):
        validate_action_dist(np.full((2, 2), 0.3), 2, 2, 1)


# ------------------------------------------------------ integration (sample)
@needs_sample
def test_item_context_alignment_on_real_files():
    r = read_item_context(SAMPLE, "random", "men")
    b = read_item_context(SAMPLE, "bts", "men")
    assert not (r["item_feature_1"] == b["item_feature_1"]).all()  # hashes really differ
    aligned = align_item_context(r, b)
    assert aligned.equals(r)


@needs_sample
def test_build_and_load_sample(tmp_path):
    cfg = DataConfig(sample_dir=str(SAMPLE), processed_dir=str(tmp_path), use_sample=True, campaigns=["men"])
    meta = build_campaign(cfg, "men")
    assert meta["splits"]["random"]["train"]["n_rounds"] == 7000
    assert meta["splits"]["bts"]["test"]["n_rounds"] == 3000

    tr = load_bandit_feedback("men", "random", "train", tmp_path)
    te = load_bandit_feedback("men", "random", "test", tmp_path)
    bts = load_bandit_feedback("men", "bts", "test", tmp_path)
    for bf in (tr, te, bts):
        assert bf["context"].shape == (bf["n_rounds"], meta["context_dim"])
        assert bf["context"].dtype == np.float32 and np.isfinite(bf["context"]).all()
        assert bf["action"].max() < 34 and set(np.unique(bf["position"])) <= {0, 1, 2}
        assert ((bf["pscore"] > 0) & (bf["pscore"] <= 1)).all()
        assert bf["action_context"].shape[0] == 34
    assert tr["timestamp"].max() <= te["timestamp"].min()  # no time leakage
    assert np.allclose(tr["pscore"], 1 / 34)
    gt = on_policy_value(bts["reward"], n_bootstrap=100)
    assert gt["ci_low"] <= gt["value"] <= gt["ci_high"]
