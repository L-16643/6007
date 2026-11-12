"""Week-1 OPE demo: compare DM / IPS / SNIPS / DR with a known policy value.

The demo builds a small contextual-bandit dataset with a known reward function,
fits an out-of-fold click model, and evaluates a few target policies. When the
official Open Bandit Pipeline (OBP) is available, the same inputs are also passed
to OBP's estimators so that the two implementations can be compared numerically.

Typical usage (the repository's conda environment is named openbandit):

    conda run -n openbandit python ope_demo.py

A dependency-free local generator is available for machines without OBP:

    python ope_demo.py --dataset local

The script writes ope_demo_results.csv and ope_demo_summary.json under
results/week1 by default.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ope.estimators import evaluate
from ope.reward_model import cross_fit

N_ACTIONS = 3
DIM_CONTEXT = 2
POLICIES = ("behavior", "uniform", "contextual", "sharp")
ESTIMATORS = ("DM", "IPS", "SNIPS", "DR")

# Data-generating process. Coefficients are fixed so that every run with the
# same seed is reproducible and the target policies genuinely differ from the
# behavior policy.
REWARD_COEF = np.array(
    [
        [1.10, -0.65, 0.35],
        [-0.40, 0.90, -0.55],
    ]
)
REWARD_INTERCEPT = np.array([-0.30, 0.05, 0.40])
BEHAVIOR_COEF = np.array(
    [
        [0.55, -0.45, -0.10],
        [-0.20, 0.35, 0.05],
    ]
)
BEHAVIOR_INTERCEPT = np.array([0.00, -0.05, 0.05])
EVAL_COEF = np.array(
    [
        [-0.80, 1.20, -0.40],
        [0.50, -0.90, 1.10],
    ]
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax with a temperature parameter."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def _reward_function(
    context: np.ndarray,
    action_context: np.ndarray | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Known expected reward q(x, a) used by the synthetic dataset.

    action_context and random_state are accepted because OBP passes them to a
    user-supplied reward function.
    """
    del action_context, random_state
    return _sigmoid(context @ REWARD_COEF + REWARD_INTERCEPT)


def _behavior_probs(context: np.ndarray) -> np.ndarray:
    """Context-dependent behavior policy with full support."""
    return _softmax(context @ BEHAVIOR_COEF + BEHAVIOR_INTERCEPT, temperature=1.15)


def _behavior_policy_function(
    context: np.ndarray,
    action_context: np.ndarray | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """OBP callback for the behavior policy."""
    del action_context, random_state
    return _behavior_probs(context)


def _target_probs(context: np.ndarray, name: str) -> np.ndarray:
    """Return a (n_rounds, n_actions) target-policy distribution."""
    if name == "behavior":
        return _behavior_probs(context)
    if name == "uniform":
        return np.full((context.shape[0], N_ACTIONS), 1.0 / N_ACTIONS)
    logits = context @ EVAL_COEF
    if name == "contextual":
        return _softmax(logits, temperature=1.0)
    if name == "sharp":
        return _softmax(logits, temperature=0.35)
    raise ValueError(f"unknown target policy: {name}")


def _build_obp_feedback(n_rounds: int, seed: int) -> dict[str, Any]:
    """Generate feedback with OBP's SyntheticBanditDataset."""
    try:
        from obp.dataset import SyntheticBanditDataset
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "Open Bandit Pipeline is not installed; use --dataset local or "
            "install it in the openbandit conda environment"
        ) from exc

    dataset = SyntheticBanditDataset(
        n_actions=N_ACTIONS,
        dim_context=DIM_CONTEXT,
        reward_type="binary",
        reward_function=_reward_function,
        behavior_policy_function=_behavior_policy_function,
        random_state=seed,
    )
    feedback = dict(dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds))
    # SyntheticBanditDataset does not model multiple recommendation slots.
    feedback["position"] = np.zeros(n_rounds, dtype=int)
    feedback["len_list"] = 1
    return feedback


def _build_local_feedback(n_rounds: int, seed: int) -> dict[str, Any]:
    """Dependency-free generator with the same data-generating process."""
    rng = np.random.default_rng(seed)
    context = rng.normal(size=(n_rounds, DIM_CONTEXT))
    behavior_probs = _behavior_probs(context)
    draws = rng.random(n_rounds)
    action = (np.cumsum(behavior_probs, axis=1) < draws[:, None]).sum(axis=1)
    action = np.clip(action, 0, N_ACTIONS - 1)

    expected_reward = _reward_function(context)
    rows = np.arange(n_rounds)
    reward = rng.binomial(1, expected_reward[rows, action])
    pscore = behavior_probs[rows, action]
    return {
        "n_rounds": n_rounds,
        "n_actions": N_ACTIONS,
        "context": context,
        "action_context": np.eye(N_ACTIONS, dtype=int),
        "action": action.astype(int),
        "position": np.zeros(n_rounds, dtype=int),
        "reward": reward.astype(int),
        "expected_reward": expected_reward,
        "pscore": pscore,
        "len_list": 1,
    }


def build_feedback(dataset: str, n_rounds: int, seed: int) -> dict[str, Any]:
    """Build and validate a synthetic bandit-feedback dictionary."""
    if n_rounds <= 0:
        raise ValueError("n_rounds must be positive")
    if dataset == "obp":
        feedback = _build_obp_feedback(n_rounds, seed)
    elif dataset == "local":
        feedback = _build_local_feedback(n_rounds, seed)
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    feedback["n_actions"] = int(feedback["n_actions"])
    feedback["len_list"] = int(feedback["len_list"])
    feedback["n_rounds"] = int(feedback["n_rounds"])

    required = ("context", "action", "position", "reward", "pscore", "expected_reward")
    for key in required:
        if key not in feedback:
            raise ValueError(f"feedback is missing {key!r}")
    n = len(feedback["reward"])
    if feedback["expected_reward"].shape != (n, feedback["n_actions"]):
        raise ValueError("expected_reward must have shape (n_rounds, n_actions)")
    if not np.all((feedback["pscore"] > 0) & (feedback["pscore"] <= 1)):
        raise ValueError("pscore must lie in (0, 1]")
    return feedback


def _official_estimates(
    feedback: dict[str, Any], action_dist: np.ndarray, q_hat: np.ndarray
) -> dict[str, float]:
    """Evaluate the same inputs with OBP's reference estimators."""
    from obp.ope import (
        DirectMethod,
        DoublyRobust,
        InverseProbabilityWeighting,
        SelfNormalizedInverseProbabilityWeighting,
    )

    common = dict(
        reward=feedback["reward"],
        action=feedback["action"],
        pscore=feedback["pscore"],
        action_dist=action_dist,
        position=feedback["position"],
    )
    return {
        "DM": float(
            DirectMethod().estimate_policy_value(
                action_dist=action_dist,
                estimated_rewards_by_reg_model=q_hat,
                position=feedback["position"],
            )
        ),
        "IPS": float(InverseProbabilityWeighting().estimate_policy_value(**common)),
        "SNIPS": float(
            SelfNormalizedInverseProbabilityWeighting().estimate_policy_value(**common)
        ),
        "DR": float(
            DoublyRobust().estimate_policy_value(
                **common, estimated_rewards_by_reg_model=q_hat
            )
        ),
    }


def _true_value(feedback: dict[str, Any], action_dist: np.ndarray) -> float:
    """Known policy value E_x[sum_a pi_e(a|x) q(x,a)] for the synthetic DGP."""
    return float(
        np.mean(np.sum(action_dist * np.asarray(feedback["expected_reward"]), axis=1))
    )


def _write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ope_demo_results.csv"
    fieldnames = [
        "dataset",
        "n_rounds",
        "seed",
        "policy",
        "estimator",
        "implementation",
        "estimate",
        "true_value",
        "abs_error",
        "squared_error",
        "relative_error_pct",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "ope_demo_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        choices=("obp", "local"),
        default="obp",
        help="Use OBP SyntheticBanditDataset or the local dependency-free DGP.",
    )
    parser.add_argument("--n-rounds", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3, help="Cross-fitting folds.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/week1"),
        help="Directory for the CSV and JSON artifacts.",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Print results without writing files."
    )
    args = parser.parse_args()
    if args.n_rounds <= 0:
        parser.error("--n-rounds must be positive")
    if args.folds < 2:
        parser.error("--folds must be at least 2")

    feedback = build_feedback(args.dataset, args.n_rounds, args.seed)
    q_hat = cross_fit(feedback, folds=args.folds, seed=args.seed)

    try:
        import obp  # noqa: F401

        obp_available = True
    except ImportError:
        obp_available = False

    rows: list[dict[str, Any]] = []
    policy_summary: dict[str, Any] = {}
    max_diff = 0.0
    print(
        f"dataset={args.dataset} n_rounds={args.n_rounds} seed={args.seed} "
        f"folds={args.folds} obp={'yes' if obp_available else 'no'}"
    )
    print(
        f"{'policy':10s} {'est':5s} {'truth':>10s} {'ours':>10s} "
        f"{'OBP':>10s} {'|diff|':>10s} {'abs err':>10s}"
    )

    for policy in POLICIES:
        action_dist_2d = _target_probs(feedback["context"], policy)
        action_dist = action_dist_2d[:, :, None]
        truth = _true_value(feedback, action_dist_2d)
        ours = evaluate(feedback, action_dist, q_hat)
        official = (
            _official_estimates(feedback, action_dist, q_hat) if obp_available else {}
        )

        estimates: dict[str, Any] = {}
        for name in ESTIMATORS:
            value = float(ours[name])
            estimates[name] = {"ours": value}
            rows.append(
                {
                    "dataset": args.dataset,
                    "n_rounds": args.n_rounds,
                    "seed": args.seed,
                    "policy": policy,
                    "estimator": name,
                    "implementation": "ours",
                    "estimate": value,
                    "true_value": truth,
                    "abs_error": abs(value - truth),
                    "squared_error": (value - truth) ** 2,
                    "relative_error_pct": 100.0 * abs(value - truth) / truth,
                }
            )
            if name in official:
                ref = float(official[name])
                diff = abs(value - ref)
                max_diff = max(max_diff, diff)
                estimates[name]["obp"] = ref
                estimates[name]["abs_diff_vs_obp"] = diff
                rows.append(
                    {
                        "dataset": args.dataset,
                        "n_rounds": args.n_rounds,
                        "seed": args.seed,
                        "policy": policy,
                        "estimator": name,
                        "implementation": "obp",
                        "estimate": ref,
                        "true_value": truth,
                        "abs_error": abs(ref - truth),
                        "squared_error": (ref - truth) ** 2,
                        "relative_error_pct": 100.0 * abs(ref - truth) / truth,
                    }
                )
            ref_text = f"{official[name]:10.6f}" if name in official else f"{'n/a':>10s}"
            diff_text = (
                f"{abs(value - official[name]):10.2e}"
                if name in official
                else f"{'-':>10s}"
            )
            print(
                f"{policy:10s} {name:5s} {truth:10.6f} {value:10.6f} "
                f"{ref_text} {diff_text} {abs(value - truth):10.2e}"
            )
        policy_summary[policy] = {
            "true_value": truth,
            "estimates": estimates,
        }

    summary = {
        "config": {
            "dataset": args.dataset,
            "n_rounds": args.n_rounds,
            "seed": args.seed,
            "folds": args.folds,
            "n_actions": N_ACTIONS,
            "estimators": list(ESTIMATORS),
            "policies": list(POLICIES),
        },
        "obp_available": obp_available,
        "max_abs_diff_vs_obp": max_diff,
        "policies": policy_summary,
    }

    if not args.no_save:
        csv_path, json_path = _write_outputs(args.output_dir, rows, summary)
        print(f"\nsaved: {csv_path}")
        print(f"saved: {json_path}")
    if obp_available:
        print(f"max |ours - OBP| = {max_diff:.3e}")


if __name__ == "__main__":
    main()
