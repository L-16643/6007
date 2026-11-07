"""End-to-end data pipeline: raw OBD csv -> cleaned, split, encoded arrays.

Output layout (one folder per campaign)::

    data/processed/<campaign>/
        meta.json                 # shapes, feature names, split boundaries, cleaning report
        encoder.json              # fitted ContextEncoder (to encode new data identically)
        action_context.npy        # (n_actions, d_a) item features, shared by both policies
        <policy>_<split>/         # policy in {random, bts}, split in {train, val, test}
            context.npy           # (n, d_x) float32
            action.npy            # (n,)  int64   item id 0..n_actions-1
            position.npy          # (n,)  int64   slot 0..2
            reward.npy            # (n,)  int64   click 0/1
            pscore.npy            # (n,)  float64 behaviour propensity pi_b(a|x, pos)
            timestamp.npy         # (n,)  int64   ns since epoch (UTC)

``load_bandit_feedback`` returns the same dict structure as
``obp.dataset.OpenBanditDataset.obtain_batch_bandit_feedback`` so the arrays
can be passed straight into obp estimators if a teammate wants to.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .io import (
    CAMPAIGNS,
    LEN_LIST,
    N_ITEMS,
    POLICIES,
    affinity_cols,
    align_item_context,
    clean_logs,
    read_item_context,
    read_logs,
)
from .features import ContextEncoder, build_action_context

log = logging.getLogger(__name__)
SPLITS = ("train", "val", "test")
ARRAYS = ("context", "action", "position", "reward", "pscore", "timestamp")


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    sample_dir: str = "data/sample"
    processed_dir: str = "data/processed"
    use_sample: bool = False
    campaigns: list[str] = field(default_factory=lambda: list(CAMPAIGNS))
    policies: list[str] = field(default_factory=lambda: list(POLICIES))
    max_rows: int | None = None
    seed: int = 12345
    val_size: float = 0.0
    test_size: float = 0.3
    min_category_freq: int = 10
    affinity_transform: str = "log1p"
    add_affinity_stats: bool = True
    standardize: bool = True

    @property
    def source_dir(self) -> str:
        return self.sample_dir if self.use_sample else self.raw_dir

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "DataConfig":
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        flat = {**cfg.get("data", {}), **cfg.get("split", {}), **cfg.get("features", {})}
        flat.update({k: v for k, v in overrides.items() if v is not None})
        known = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in flat.items() if k in known})


# ----------------------------------------------------------------------------
# splitting
# ----------------------------------------------------------------------------
def time_split_bounds(n: int, val_size: float, test_size: float) -> dict[str, tuple[int, int]]:
    """Chronological split: earliest rows -> train, then val, latest -> test."""
    if not (0 <= val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1):
        raise ValueError("need 0 <= val_size, 0 < test_size, val_size + test_size < 1")
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_val - n_test
    b = {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, n)}
    return {k: v for k, v in b.items() if v[1] > v[0]}


# ----------------------------------------------------------------------------
# building
# ----------------------------------------------------------------------------
def _to_arrays(df: pd.DataFrame, encoder: ContextEncoder) -> dict[str, np.ndarray]:
    return {
        "context": encoder.transform(df),
        "action": df["item_id"].to_numpy(dtype=np.int64),
        "position": df["position"].to_numpy(dtype=np.int64),
        "reward": df["click"].to_numpy(dtype=np.int64),
        "pscore": df["propensity_score"].to_numpy(dtype=np.float64),
        "timestamp": df["timestamp"].astype("int64").to_numpy(),
    }


def build_campaign(cfg: DataConfig, campaign: str) -> dict:
    """Process one campaign for all configured policies and write to disk."""
    t0 = time.time()
    out_dir = Path(cfg.processed_dir) / campaign
    out_dir.mkdir(parents=True, exist_ok=True)
    src = cfg.source_dir

    # ---- read + clean every policy
    frames, reports = {}, {}
    for policy in cfg.policies:
        df = read_logs(src, policy, campaign, max_rows=cfg.max_rows)
        df, rep = clean_logs(df, campaign)
        frames[policy], reports[policy] = df, rep
        log.info("%s/%s cleaned: %d -> %d rows, CTR %.5f", policy, campaign, rep["n_raw"], rep["n_clean"], rep["ctr"])

    # ---- item context: random folder is canonical, others are verified + aligned
    canon_policy = "random" if "random" in cfg.policies else cfg.policies[0]
    canon_ic = read_item_context(src, canon_policy, campaign)
    for policy in cfg.policies:
        if policy != canon_policy:
            align_item_context(canon_ic, read_item_context(src, policy, campaign))
    action_context, action_feature_names = build_action_context(canon_ic)
    np.save(out_dir / "action_context.npy", action_context)

    # ---- split chronologically, per policy
    bounds = {p: time_split_bounds(len(df), cfg.val_size, cfg.test_size) for p, df in frames.items()}

    # ---- fit encoder on the canonical policy's TRAIN split only.
    # User contexts are not affected by the logging policy, so random-train is
    # an unbiased sample of the context distribution; fitting on train only
    # avoids leaking test statistics.
    s, e = bounds[canon_policy]["train"]
    encoder = ContextEncoder(
        affinity_cols=affinity_cols(campaign),
        min_category_freq=cfg.min_category_freq,
        affinity_transform=cfg.affinity_transform,
        add_affinity_stats=cfg.add_affinity_stats,
        standardize=cfg.standardize,
    ).fit(frames[canon_policy].iloc[s:e])
    with open(out_dir / "encoder.json", "w", encoding="utf-8") as f:
        json.dump(encoder.to_dict(), f, ensure_ascii=False, indent=1)

    # ---- encode + save each (policy, split)
    split_info: dict = {}
    for policy in list(frames):
        df = frames[policy]
        split_info[policy] = {}
        for split, (s, e) in bounds[policy].items():
            part = df.iloc[s:e]
            arrays = _to_arrays(part, encoder)
            d = out_dir / f"{policy}_{split}"
            d.mkdir(exist_ok=True)
            for k in ARRAYS:
                np.save(d / f"{k}.npy", arrays[k])
            split_info[policy][split] = {
                "n_rounds": int(e - s),
                "n_clicks": int(arrays["reward"].sum()),
                "ctr": float(arrays["reward"].mean()),
                "time_start": str(part["timestamp"].iloc[0]),
                "time_end": str(part["timestamp"].iloc[-1]),
            }
            del arrays
        del frames[policy], df  # free memory before the next policy

    meta = {
        "campaign": campaign,
        "source": "sample" if cfg.use_sample else "full",
        "n_actions": N_ITEMS[campaign],
        "len_list": LEN_LIST,
        "context_dim": len(encoder.feature_names),
        "context_feature_names": encoder.feature_names,
        "action_context_dim": int(action_context.shape[1]),
        "action_context_feature_names": action_feature_names,
        "policies": list(cfg.policies),
        "splits": split_info,
        "cleaning": reports,
        "config": cfg.__dict__,
        "build_seconds": round(time.time() - t0, 1),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=str)
    log.info("campaign %s done in %.1fs -> %s", campaign, meta["build_seconds"], out_dir)
    return meta


def build_all(cfg: DataConfig) -> dict[str, dict]:
    return {c: build_campaign(cfg, c) for c in cfg.campaigns}


# ----------------------------------------------------------------------------
# loading (API for teammates)
# ----------------------------------------------------------------------------
def load_meta(campaign: str, processed_dir: str | Path = "data/processed") -> dict:
    with open(Path(processed_dir) / campaign / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def load_encoder(campaign: str, processed_dir: str | Path = "data/processed") -> ContextEncoder:
    with open(Path(processed_dir) / campaign / "encoder.json", encoding="utf-8") as f:
        return ContextEncoder.from_dict(json.load(f))


def load_bandit_feedback(
    campaign: str = "all",
    policy: str = "random",
    split: str = "train",
    processed_dir: str | Path = "data/processed",
    mmap: bool = False,
) -> dict:
    """Load one (campaign, policy, split) as an obp-style ``bandit_feedback`` dict.

    Keys: ``n_rounds, n_actions, len_list, action, position, reward, pscore,
    context, action_context, timestamp, campaign, policy, split``.

    ``mmap=True`` memory-maps ``context`` (useful for bts/all, ~8M rows).
    """
    base = Path(processed_dir) / campaign
    d = base / f"{policy}_{split}"
    if not d.exists():
        raise FileNotFoundError(f"{d} not found — run scripts/prepare_data.py first")
    meta = load_meta(campaign, processed_dir)
    bf = {k: np.load(d / f"{k}.npy", mmap_mode="r" if (mmap and k == "context") else None) for k in ARRAYS}
    bf.update(
        n_rounds=int(bf["action"].shape[0]),
        n_actions=int(meta["n_actions"]),
        len_list=int(meta["len_list"]),
        action_context=np.load(base / "action_context.npy"),
        campaign=campaign,
        policy=policy,
        split=split,
    )
    return bf
