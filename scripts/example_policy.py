"""Example for member C: produce an action_dist and get its Ground Truth.

Builds two toy policies on random/train and evaluates them on random/test:
  * uniform     — must reproduce the on-policy CTR of Random exactly
  * popularity  — softmax over per-(position, item) CTR learned on train

    uv run python scripts/example_policy.py --campaign men
"""

import argparse

import numpy as np

from ope.data import load_bandit_feedback, policy_value_from_random_logs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="men")
    ap.add_argument("--temperature", type=float, default=0.002)
    args = ap.parse_args()

    tr = load_bandit_feedback(args.campaign, "random", "train")
    te = load_bandit_feedback(args.campaign, "random", "test")
    K, L, n = te["n_actions"], te["len_list"], te["n_rounds"]

    # per-(item, position) CTR on train, with add-one smoothing toward the global CTR
    clicks = np.zeros((K, L)); shows = np.zeros((K, L))
    np.add.at(clicks, (tr["action"], tr["position"]), tr["reward"])
    np.add.at(shows, (tr["action"], tr["position"]), 1)
    ctr = (clicks + tr["reward"].mean() * 100) / (shows + 100)
    logits = ctr / args.temperature
    probs = np.exp(logits - logits.max(0)); probs /= probs.sum(0)          # (K, L)

    policies = {
        "uniform": np.full((n, K, L), 1.0 / K),
        "popularity": np.broadcast_to(probs, (n, K, L)),                  # context-free example
    }
    print(f"campaign={args.campaign}  on-policy Random CTR (test) = {te['reward'].mean():.6f}")
    for name, ad in policies.items():
        r = policy_value_from_random_logs(te, ad, n_bootstrap=500)
        print(f"{name:11s} V_GT={r['value']:.6f}  95%CI=[{r['ci_low']:.6f}, {r['ci_high']:.6f}]  "
              f"mean_w={r['mean_weight']:.3f}  max_w={r['max_weight']:.1f}  ESS={r['ess']:.0f}")


if __name__ == "__main__":
    main()
