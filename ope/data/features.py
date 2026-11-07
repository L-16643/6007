"""Feature engineering for user contexts and item (action) contexts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .io import ITEM_CAT_COLS, ITEM_NUM_COLS, USER_CAT_COLS

OTHER = "__other__"


@dataclass
class ContextEncoder:
    """Turns raw user columns into a dense float32 context matrix ``X``.

    Layout of ``X`` (column order is stored in ``feature_names``)::

        [ one-hot(user_feature_0..3, rare -> __other__)
        | transformed user-item affinities (K cols)
        | affinity stats: log1p(total affinity), #items with affinity > 0 ]

    One-hot columns are left as 0/1; continuous columns are z-scored with
    statistics fitted on the training split only.
    """

    affinity_cols: list[str]
    min_category_freq: int = 10
    affinity_transform: str = "log1p"  # "none" | "log1p"
    add_affinity_stats: bool = True
    standardize: bool = True
    # fitted state
    vocab: dict[str, list[str]] = field(default_factory=dict)
    mean_: list[float] | None = None
    std_: list[float] | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "ContextEncoder":
        self.vocab = {}
        for c in USER_CAT_COLS:
            counts = df[c].value_counts()
            keep = [str(k) for k in counts[counts >= self.min_category_freq].index]
            self.vocab[c] = sorted(keep) + [OTHER]
        if self.standardize:
            # streaming mean / variance so fitting on millions of rows stays cheap
            s1 = s2 = None
            n = 0
            for s in range(0, len(df), 500_000):
                cont = self._continuous(df.iloc[s : s + 500_000]).astype(np.float64)
                s1 = cont.sum(0) if s1 is None else s1 + cont.sum(0)
                s2 = (cont**2).sum(0) if s2 is None else s2 + (cont**2).sum(0)
                n += len(cont)
            mu = s1 / n
            sd = np.sqrt(np.maximum(s2 / n - mu**2, 0.0))
            sd[sd < 1e-8] = 1.0  # constant columns stay constant
            self.mean_, self.std_ = mu.tolist(), sd.tolist()
        return self

    # ------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame, chunk_size: int = 500_000) -> np.ndarray:
        if not self.vocab:
            raise RuntimeError("ContextEncoder is not fitted")
        n = len(df)
        out = np.zeros((n, len(self.feature_names)), dtype=np.float32)

        # one-hot via category codes (fast on millions of rows)
        offset = 0
        rows = np.arange(n)
        for c in USER_CAT_COLS:
            voc = self.vocab[c]
            index = {v: i for i, v in enumerate(voc)}
            cat = df[c].astype("category")
            lut = np.array([index.get(str(v), index[OTHER]) for v in cat.cat.categories] + [index[OTHER]])
            codes = lut[cat.cat.codes.to_numpy()]  # code -1 (missing) -> last lut slot -> __other__
            out[rows, offset + codes] = 1.0
            offset += len(voc)

        # continuous block, chunked to bound peak memory
        if self.standardize:
            mu = np.asarray(self.mean_, dtype=np.float32)
            sd = np.asarray(self.std_, dtype=np.float32)
        for s in range(0, n, chunk_size):
            cont = self._continuous(df.iloc[s : s + chunk_size])
            if self.standardize:
                cont = (cont - mu) / sd
            out[s : s + chunk_size, offset:] = cont
        return out

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # --------------------------------------------------------------- helpers
    def _continuous(self, df: pd.DataFrame) -> np.ndarray:
        raw = df[self.affinity_cols].to_numpy(dtype=np.float32)
        parts = [np.log1p(raw) if self.affinity_transform == "log1p" else raw]
        if self.add_affinity_stats:
            parts.append(np.log1p(raw.sum(axis=1, keepdims=True)))
            parts.append((raw > 0).sum(axis=1, keepdims=True).astype(np.float32))
        return np.hstack(parts)

    @property
    def n_onehot(self) -> int:
        return sum(len(v) for v in self.vocab.values())

    @property
    def feature_names(self) -> list[str]:
        names = [f"{c}={v[:8]}" for c in USER_CAT_COLS for v in self.vocab[c]]
        names += [f"aff_{c.rsplit('_', 1)[-1]}" for c in self.affinity_cols]
        if self.add_affinity_stats:
            names += ["aff_log_total", "aff_n_nonzero"]
        return names

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ContextEncoder":
        return cls(**d)


def build_action_context(item_context: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Item feature matrix of shape ``(n_actions, d_a)``, row ``a`` = item ``a``.

    ``item_feature_0`` is numeric (already standardised in the release); the
    three categorical features are one-hot encoded.
    """
    ic = item_context.sort_values("item_id").reset_index(drop=True)
    if not np.array_equal(ic["item_id"].to_numpy(), np.arange(len(ic))):
        raise ValueError("item_id must be 0..n_actions-1")
    blocks = [ic[ITEM_NUM_COLS].to_numpy(dtype=np.float32)]
    names = list(ITEM_NUM_COLS)
    for c in ITEM_CAT_COLS:
        d = pd.get_dummies(ic[c], prefix=c, dtype=np.float32)
        d = d[sorted(d.columns)]
        blocks.append(d.to_numpy())
        names += [n[: len(c) + 9] for n in d.columns]
    return np.hstack(blocks), names
