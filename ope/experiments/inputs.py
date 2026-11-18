"""Validate member B/C artifacts and construct independent Random references."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ope.data import load_bandit_feedback, load_meta
from ope.data.ground_truth import bootstrap_mean_ci

POLICIES = ("linucb", "logisticts")
BLOCK_SIZE = 8192


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_digest(feedback):
    digest = hashlib.sha256()
    for start in range(0, feedback["n_rounds"], BLOCK_SIZE):
        keys = np.stack(
            [feedback[key][start:start + BLOCK_SIZE]
             for key in ("timestamp", "action", "position")], axis=1
        )
        digest.update(keys.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def validate_distribution(distribution, feedback):
    shape = (feedback["n_rounds"], feedback["n_actions"], feedback["len_list"])
    if distribution.shape not in (shape, shape[:2]):
        raise ValueError(f"Policy shape {distribution.shape} differs from {shape}")
    for start in range(0, shape[0], BLOCK_SIZE):
        block = distribution[start:start + BLOCK_SIZE]
        if (not np.isfinite(block).all() or np.any(block < 0)
                or not np.allclose(block.sum(axis=1), 1, atol=1e-6, rtol=0)):
            raise ValueError(f"Invalid policy probabilities at row {start}")


def timestamp_ns(values, split_meta):
    """Infer historical integer timestamp units from recorded UTC endpoints."""
    values = np.asarray(values)
    if values.ndim != 1 or not values.size or values.dtype.kind not in "iu":
        raise ValueError("Expected nonempty integer timestamps")
    if np.any(values[1:] < values[:-1]):
        raise ValueError("Timestamps must be chronologically ordered")
    endpoints = [pd.Timestamp(split_meta[key]).value for key in ("time_start", "time_end")]
    candidates = []
    for factor in (1, 1000, 1_000_000, 1_000_000_000):
        converted = [int(values[0]) * factor, int(values[-1]) * factor]
        if converted == endpoints:
            candidates.append(factor)
    if len(candidates) != 1:
        raise ValueError("Timestamp units cannot be uniquely matched to metadata")
    factor = candidates[0]
    if max(abs(int(values[0])), abs(int(values[-1]))) > np.iinfo(np.int64).max // factor:
        raise ValueError("Timestamp conversion overflows int64")
    return values.astype(np.int64) * factor


def load_inputs(campaign, processed_dir, policy_dir):
    directory = Path(policy_dir) / campaign
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    meta = load_meta(campaign, processed_dir)
    expected = (manifest["campaign"] == campaign
                and manifest["source"] == meta["source"]
                and manifest["data_splits"] == meta["splits"]
                and manifest["context_feature_names"] == meta["context_feature_names"])
    if not expected:
        raise ValueError(f"{campaign}: policy manifest and processed data differ")
    logs, distributions = {}, {}
    paths = [directory / "manifest.json", Path(processed_dir) / campaign / "meta.json"]
    for name in ("random", "bts"):
        feedback = load_bandit_feedback(campaign, name, "test", processed_dir, mmap=True)
        n = feedback["n_rounds"]
        for key in ("action", "position", "reward", "pscore", "timestamp"):
            if feedback[key].shape != (n,):
                raise ValueError(f"{campaign}/{name}: misaligned {key}")
        for key, size in (("action", feedback["n_actions"]), ("position", feedback["len_list"])):
            vector = feedback[key]
            if vector.dtype.kind not in "iu" or np.any((vector < 0) | (vector >= size)):
                raise ValueError(f"{campaign}/{name}: invalid {key}")
        if (not np.isin(feedback["reward"], [0, 1]).all()
                or not np.all((feedback["pscore"] > 0) & (feedback["pscore"] <= 1))):
            raise ValueError(f"{campaign}/{name}: invalid clicks or propensities")
        if name == "random" and not np.allclose(feedback["pscore"], 1 / feedback["n_actions"], rtol=1e-6):
            raise ValueError("Random reference requires uniform propensities")
        test = manifest["test"][name]
        if row_digest(feedback) != test["row_keys_sha256"]:
            raise ValueError(f"{campaign}/{name}: row order differs from policy export")
        distributions[name] = {}
        for policy in POLICIES:
            path = directory / test["files"][policy]
            distribution = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(distribution.shape) != test["shape"]:
                raise ValueError(f"{campaign}/{name}: manifest shape mismatch")
            validate_distribution(distribution, feedback)
            distributions[name][policy] = distribution
            paths.append(path)
        base = Path(processed_dir) / campaign / f"{name}_test"
        paths.extend(base / f"{key}.npy" for key in
                     ("context", "action", "position", "reward", "pscore", "timestamp"))
        for start in range(0, n, BLOCK_SIZE):
            if not np.isfinite(feedback["context"][start:start + BLOCK_SIZE]).all():
                raise ValueError(f"{campaign}/{name}: nonfinite context")
        logs[name] = feedback
    train_path = Path(processed_dir) / campaign / "random_train" / "timestamp.npy"
    paths.append(train_path)
    return dict(campaign=campaign, meta=meta, export_manifest=manifest, logs=logs, distributions=distributions,
                training_timestamps=np.load(train_path, mmap_mode="r"),
                fingerprints={str(path.resolve()): file_digest(path) for path in paths},
                provenance_status="export_row_keys_and_features_verified; current_files_sha256")


def select_test_rows(inputs, window):
    meta = inputs["meta"]
    times = {name: timestamp_ns(log["timestamp"], meta["splits"][name]["test"])
             for name, log in inputs["logs"].items()}
    train_times = timestamp_ns(inputs["training_timestamps"], meta["splits"]["random"]["train"])
    start = max(int(values[0]) for values in times.values())
    stop = min(int(values[-1]) for values in times.values())
    rows = {}
    for name, values in times.items():
        if window == "intersection":
            rows[name] = np.flatnonzero((values >= start) & (values <= stop) & (values > train_times[-1]))
        elif window == "native":
            rows[name] = np.arange(len(values))
        else:
            raise ValueError("Unknown test window")
        if rows[name].size < 2:
            raise ValueError(f"{inputs['campaign']}/{name}: fewer than two rows in {window} window")
    bounds = {name: {"start_utc": pd.Timestamp(times[name][index[0]], tz="UTC").isoformat(),
                     "end_utc": pd.Timestamp(times[name][index[-1]], tz="UTC").isoformat(),
                     "n_rounds": len(index),
                     "position_fraction": np.bincount(inputs["logs"][name]["position"][index],
                                                     minlength=meta["len_list"]).tolist()}
              for name, index in rows.items()}
    for info in bounds.values():
        info["position_fraction"] = [count / info["n_rounds"] for count in info["position_fraction"]]
    window_id = hashlib.sha256(json.dumps(bounds, sort_keys=True).encode()).hexdigest()[:16]
    return rows, dict(mode=window, window_id=window_id, logs=bounds)


def logged_weights(feedback, distribution):
    rows = np.arange(len(feedback["reward"]))
    if distribution.ndim == 2:
        probability = distribution[rows, feedback["action"]]
    else:
        probability = distribution[rows, feedback["action"], feedback["position"]]
    return probability.astype(np.float64) / feedback["pscore"]


def compute_references(inputs, random_rows, *, bootstrap_repeats, seed):
    feedback = inputs["logs"]["random"]
    contributions, records = {}, []
    for number, policy in enumerate(POLICIES):
        weights = np.empty(len(random_rows))
        for start in range(0, len(random_rows), BLOCK_SIZE):
            index = random_rows[start:start + BLOCK_SIZE]
            part = {key: feedback[key][index] for key in ("action", "position", "reward", "pscore")}
            weights[start:start + len(index)] = logged_weights(
                part, inputs["distributions"]["random"][policy][index])
        values = weights * feedback["reward"][random_rows]
        contributions[policy] = values
        value = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
        low, high, _ = bootstrap_mean_ci(values, bootstrap_repeats, seed=seed + number)
        total = weights.sum()
        records.append(dict(
            campaign=inputs["campaign"], policy=policy, source=inputs["meta"]["source"],
            value=value, se=se, normal_ci_low=value - norm.ppf(.975) * se,
            normal_ci_high=value + norm.ppf(.975) * se,
            bootstrap_ci_low=low, bootstrap_ci_high=high, n_rounds=len(values),
            mean_weight=float(weights.mean()), max_weight=float(weights.max()),
            ess=float(total ** 2 / np.square(weights).sum()) if total else 0.,
            positive_weight_count=int(np.count_nonzero(weights)),
            positive_contribution_count=int(np.count_nonzero(values)),
            reference_status="ok" if np.any(values) else "zero_contributions"))
    difference = contributions["linucb"] - contributions["logisticts"]
    low, high, _ = bootstrap_mean_ci(difference, bootstrap_repeats, seed=seed + 2)
    rank = dict(value=float(difference.mean()), ci_low=low, ci_high=high,
                status="uncertain" if low <= 0 <= high else "resolved")
    return records, rank
