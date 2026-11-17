"""Evaluate C's frozen policies on repeated BTS/test subsamples."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ope.data import CAMPAIGNS, load_bandit_feedback, load_meta
from ope.estimators import evaluate
from ope.reward_model import cross_fit


def run_campaign(campaign, args):
    directory = args.policy_dir / campaign
    manifest = json.loads((directory / "manifest.json").read_text())
    meta = load_meta(campaign, args.processed_dir)
    if manifest["campaign"] != campaign or manifest["data_splits"] != meta["splits"]:
        raise ValueError(f"{campaign}: policy manifest and processed data differ")
    log = load_bandit_feedback(campaign, "bts", "test", args.processed_dir, mmap=True)
    test = manifest["test"]["bts"]
    keys = np.stack([log[k] for k in ("timestamp", "action", "position")], axis=1)
    digest = hashlib.sha256(keys.astype("<i8", copy=False)).hexdigest()
    if digest != test["row_keys_sha256"]:
        raise ValueError(f"{campaign}: BTS/test row order differs from policy export")
    policies = {name: np.load(directory / path, mmap_mode="r")
                for name, path in test["files"].items()}
    if any(list(pi.shape) != test["shape"] for pi in policies.values()):
        raise ValueError(f"{campaign}: unexpected policy shape")
    reference = manifest["test"]["random"]["random_ips_reference"]
    if args.rows >= log["n_rounds"]:
        raise ValueError("--rows must be smaller than BTS/test for repeated subsampling")
    rng = np.random.default_rng(args.seed)
    rows = []
    for repeat in range(args.repeats):
        idx = rng.choice(log["n_rounds"], args.rows, replace=False)
        sample = {k: log[k][idx] for k in ("context", "action", "position", "reward", "pscore")}
        sample.update(n_actions=log["n_actions"], len_list=log["len_list"])
        q = cross_fit(sample, seed=args.seed + repeat)
        for name, pi in policies.items():
            values = evaluate(sample, pi[idx], q, allow_zero_mass=True)
            truth = reference[name]["value"]
            for estimator, value in values.items():
                rows.append(dict(campaign=campaign, source=manifest["source"], policy=name,
                                 estimator=estimator, seed=args.seed, n=args.rows,
                                 repeat=repeat, reference=truth, estimate=value,
                                 squared_error=(value - truth) ** 2))
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("campaign", "source", "policy", "estimator", "seed", "n", "reference"))
        groups.setdefault(key, []).append(row["squared_error"])
    for key, errors in groups.items():
        valid = np.asarray(errors)[np.isfinite(errors)]
        yield dict(zip(("campaign", "source", "policy", "estimator", "seed", "n", "reference"), key),
                   repeats=len(errors), valid_repeats=len(valid),
                   mse=valid.mean() if len(valid) else np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", nargs="+", choices=CAMPAIGNS, default=CAMPAIGNS)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--policy-dir", type=Path, default=Path("outputs/policies"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/ope_mse"))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.rows < 3 or args.repeats < 2:
        parser.error("three-fold fitting requires --rows >= 3 and MSE requires --repeats >= 2")
    rows = []
    for campaign in dict.fromkeys(args.campaign):
        rows.extend(run_campaign(campaign, args))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, records in [("replicates.csv", rows), ("summary.csv", list(summarize(rows)))]:
        with (args.out_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    main()
