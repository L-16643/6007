"""Train member C's policies on Random/train and export frozen test distributions.

From the project root: uv run python -m scripts.train_policies --campaign men
"""

import argparse
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ope.data import load_bandit_feedback, load_meta, policy_value_from_random_logs
from ope.data.io import CAMPAIGNS, POLICIES
from ope.policies import LinUCB, LogisticTS


def _context_batches(context: np.ndarray, batch_size: int) -> Iterator[tuple[slice, np.ndarray, np.ndarray]]:
    for start in range(0, len(context), batch_size):
        rows = slice(start, start + batch_size)
        block = np.ascontiguousarray(context[rows])
        if not np.isfinite(block).all():
            raise ValueError(f"Nonfinite contexts in test rows beginning at {start}")
        # Duplicate contexts have identical frozen policies. Score them once
        # per block and restore the original row order using the inverse map.
        keys = block.view(np.dtype((np.void, block.dtype.itemsize * block.shape[1]))).ravel()
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        yield rows, block[first], inverse


def _export(policies: dict, feedback: dict, log: str, out: Path, batch_size: int) -> dict:
    shape = (feedback["n_rounds"], feedback["n_actions"], feedback["len_list"])
    paths = {name: out / f"{name}_{log}_test.npy" for name in policies}
    arrays = {
        name: np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        for name, path in paths.items()
    }
    for rows, context, inverse in _context_batches(feedback["context"], batch_size):
        for name, model in policies.items():
            probs = model.predict_action_dist(context)
            if not (
                np.isfinite(probs).all() and (probs >= 0).all()
                and np.allclose(probs.sum(axis=1), 1, atol=1e-6, rtol=0)
            ):
                raise RuntimeError(f"{name} emitted invalid probabilities at rows {rows}")
            arrays[name][rows] = probs[inverse]
    for array in arrays.values():
        array.flush()

    # Fingerprint the log's row keys in source order, using a portable integer layout.
    row_keys = np.stack([feedback[k] for k in ("timestamp", "action", "position")], axis=1)
    result = {
        "shape": list(shape),
        "dtype": "float32",
        "files": {k: p.name for k, p in paths.items()},
        "row_keys_sha256": hashlib.sha256(row_keys.astype("<i8", copy=False)).hexdigest(),
    }
    if log == "random":
        result["random_ctr"] = float(feedback["reward"].mean())
        result["random_ips_reference"] = {}
        for name, array in arrays.items():
            with np.errstate(divide="ignore", invalid="ignore"):
                reference = policy_value_from_random_logs(feedback, array)
            # On very small logs, zero matching actions leave SNIPS and ESS undefined.
            result["random_ips_reference"][name] = {
                k: None if isinstance(v, float) and not np.isfinite(v) else v
                for k, v in reference.items()
            }
    return result


def _run_campaign(campaign: str, args: argparse.Namespace) -> None:
    final = args.out_dir / campaign
    if final.exists():
        raise FileExistsError(f"{final} already exists. Use a fresh --out-dir to keep prior results.")
    source_meta = load_meta(campaign, args.processed_dir)
    train = load_bandit_feedback(campaign, "random", "train", args.processed_dir, mmap=True)
    policies = {
        "linucb": LinUCB(alpha=args.linucb_alpha, ridge=args.ridge),
        "logisticts": LogisticTS(prior_precision=args.prior_precision, n_draws=args.ts_draws, seed=args.seed),
    }
    for name, model in policies.items():
        print(f"{campaign}: fitting {name} on {train['n_rounds']:,} Random/train rows", flush=True)
        model.fit(train)

    manifest = {
        "campaign": campaign,
        "source": source_meta["source"],
        "training_split": "random/train",
        "training_rows": train["n_rounds"],
        "data_splits": source_meta["splits"],
        "context_feature_names": source_meta["context_feature_names"],
        "linucb": {"alpha": args.linucb_alpha, "ridge": args.ridge},
        "logisticts": {"prior_precision": args.prior_precision, "n_draws": args.ts_draws, "seed": args.seed},
        "test": {},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{campaign}-", dir=args.out_dir) as staging:
        output = Path(staging) / campaign
        output.mkdir()
        for log in dict.fromkeys(args.test_logs):
            feedback = load_bandit_feedback(campaign, log, "test", args.processed_dir, mmap=True)
            print(f"{campaign}: exporting distributions on {feedback['n_rounds']:,} {log}/test rows", flush=True)
            manifest["test"][log] = _export(policies, feedback, log, output, args.batch_size)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        output.rename(final)
    print(f"{campaign}: saved {final}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", nargs="+", choices=CAMPAIGNS, default=list(CAMPAIGNS))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/policies"))
    parser.add_argument("--test-logs", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--linucb-alpha", type=float, default=0.02)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--prior-precision", type=float, default=1.0)
    parser.add_argument("--ts-draws", type=int, default=64)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    for campaign in dict.fromkeys(args.campaign):
        _run_campaign(campaign, args)


if __name__ == "__main__":
    main()
