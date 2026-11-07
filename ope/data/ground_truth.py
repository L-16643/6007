"""Ground-truth policy values ("绝对标尺") for the OPE benchmark.

Two kinds of ground truth are provided:

1. **On-policy value of a logged policy** (``on_policy_value``)
   OBD contains real online logs of two policies (uniform Random and
   Bernoulli TS). The empirical CTR of a policy on its own held-out test logs
   is an unbiased estimate of its true online value. This is the canonical
   OBP benchmark: estimate V(bts) off-policy from *random* logs and compare
   with the on-policy CTR of the *bts* test logs (and vice versa).

2. **Value of any new policy from uniform-random logs**
   (``policy_value_from_random_logs``)
   Because the Random policy is uniform, pi_b(a|x,pos) = 1/n_actions is known
   exactly and the importance weight pi_e/pi_b <= n_actions is bounded. The
   IPS estimate on held-out random logs is therefore unbiased with controlled
   variance, and serves as ground truth for policies trained by member C
   (LinUCB, LogisticTS, ...), which were never deployed online.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ----------------------------------------------------------------------------
# confidence intervals
# ----------------------------------------------------------------------------
def wilson_ci(n_success: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli rate (accurate for tiny CTRs)."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    p = n_success / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> tuple[float, float, np.ndarray]:
    """Exact non-parametric bootstrap of the mean, efficient for sparse values.

    Rewards are clicks (~0.4% non-zero), so instead of resampling all n rows
    we use an equivalent two-stage draw: the number K of the n resampled rows
    that land on a non-zero entry is Binomial(n, m/n); those K draws are then
    uniform over the m non-zero entries. This is distributionally identical to
    ordinary resampling but costs O(m) per replicate instead of O(n).
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    nz = values[values != 0]
    m = nz.size
    rng = np.random.default_rng(seed)
    if m == 0:
        boots = np.zeros(n_bootstrap)
    else:
        k = rng.binomial(n, m / n, size=n_bootstrap)
        boots = np.array([nz[rng.integers(0, m, size=kk)].sum() / n for kk in k])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), boots


# ----------------------------------------------------------------------------
# 1) on-policy ground truth
# ----------------------------------------------------------------------------
def on_policy_value(
    reward: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    seed: int = 12345,
) -> dict:
    """True value of the logging policy = mean reward of its own logs."""
    reward = np.asarray(reward)
    n, k = int(reward.size), int(reward.sum())
    mean = k / n if n else float("nan")
    se = float(np.sqrt(mean * (1 - mean) / n)) if n else float("nan")
    w_lo, w_hi = wilson_ci(k, n, alpha)
    b_lo, b_hi, _ = bootstrap_mean_ci(reward, n_bootstrap, alpha, seed)
    return {
        "value": float(mean),
        "se": se,
        "ci_low": w_lo,
        "ci_high": w_hi,
        "ci_method": "wilson",
        "bootstrap_ci_low": b_lo,
        "bootstrap_ci_high": b_hi,
        "n_rounds": n,
        "n_clicks": k,
    }


# ----------------------------------------------------------------------------
# 2) ground truth of a new policy from uniform random logs
# ----------------------------------------------------------------------------
def validate_action_dist(action_dist: np.ndarray, n_rounds: int, n_actions: int, len_list: int) -> np.ndarray:
    """Check / reshape an evaluation-policy distribution.

    Accepts the obp convention ``(n_rounds, n_actions, len_list)`` or a
    position-independent ``(n_rounds, n_actions)`` (broadcast to all slots).
    Each ``action_dist[i, :, pos]`` must be a probability vector.
    """
    ad = np.asarray(action_dist, dtype=np.float64)
    if ad.ndim == 2:
        ad = np.repeat(ad[:, :, None], len_list, axis=2)
    if ad.shape != (n_rounds, n_actions, len_list):
        raise ValueError(f"action_dist shape {ad.shape} != {(n_rounds, n_actions, len_list)}")
    if (ad < -1e-9).any():
        raise ValueError("action_dist has negative probabilities")
    sums = ad.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        raise ValueError(f"action_dist does not sum to 1 over actions (range {sums.min():.4f}..{sums.max():.4f})")
    return ad


def policy_value_from_random_logs(
    bandit_feedback: dict,
    action_dist: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    seed: int = 12345,
) -> dict:
    """Unbiased value of ``action_dist`` estimated on uniform-random logs.

    ``action_dist`` must be computed on exactly the contexts of
    ``bandit_feedback`` (same row order), e.g.
    ``load_bandit_feedback(campaign, "random", "test")["context"]``.
    """
    pscore = np.asarray(bandit_feedback["pscore"], dtype=np.float64)
    n_actions = int(bandit_feedback["n_actions"])
    if not np.allclose(pscore, 1.0 / n_actions, rtol=1e-6):
        raise ValueError(
            "ground truth requires uniform-random logs (pscore == 1/n_actions); "
            f"got pscore in [{pscore.min():.3g}, {pscore.max():.3g}]"
        )
    n = int(bandit_feedback["n_rounds"])
    ad = validate_action_dist(action_dist, n, n_actions, int(bandit_feedback["len_list"]))
    a, pos, r = bandit_feedback["action"], bandit_feedback["position"], bandit_feedback["reward"]

    w = ad[np.arange(n), a, pos] / pscore  # bounded by n_actions
    v = w * r
    value = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(n))
    z = stats.norm.ppf(1 - alpha / 2)
    b_lo, b_hi, _ = bootstrap_mean_ci(v, n_bootstrap, alpha, seed)
    return {
        "value": value,
        "se": se,
        "ci_low": value - z * se,
        "ci_high": value + z * se,
        "ci_method": "normal",
        "bootstrap_ci_low": b_lo,
        "bootstrap_ci_high": b_hi,
        "snips_value": float(v.sum() / w.sum()),
        "mean_weight": float(w.mean()),  # should be ~1 for a valid policy
        "max_weight": float(w.max()),
        "ess": float(w.sum() ** 2 / (w**2).sum()),  # effective sample size
        "n_rounds": n,
    }


def uniform_action_dist(n_rounds: int, n_actions: int, len_list: int) -> np.ndarray:
    return np.full((n_rounds, n_actions, len_list), 1.0 / n_actions)
