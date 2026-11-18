"""Run paired clipping/sample-size experiments on validated B/C artifacts."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from ope.experiments.inputs import POLICIES, compute_references, file_digest, load_inputs, select_test_rows
from ope.experiments.sensitivity import (
    bootstrap_metrics, run_replicate, summarize_estimates, summarize_rankings,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def identity(value):
    return hashlib.sha256(json.dumps(clean_json(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, records):
    temporary = path.with_suffix(".csv.tmp")
    pd.DataFrame(records).to_csv(temporary, index=False, encoding="utf-8", float_format="%.17g")
    temporary.replace(path)


@contextmanager
def writer_lock(directory):
    """OS file locks release automatically after a process crashes."""
    path = directory / ".writer.lock"
    with path.open("a+b") as handle:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    known, _ = parser.parse_known_args(argv)
    config = yaml.safe_load(known.config.read_text(encoding="utf-8")).get("experiments", {})
    parser.add_argument("--campaign", nargs="+", choices=("all", "men", "women"), default=config.get("campaigns", ["all", "men", "women"]))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--policy-dir", type=Path, default=Path("outputs/policies/e_full"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/sensitivity/e_full"))
    parser.add_argument("--window", choices=("intersection", "native"), default=config.get("window", "intersection"))
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=config.get("sample_sizes", [1000, 2000, 5000, 10000, 20000]))
    parser.add_argument("--repeats", type=int, default=config.get("repeats", 50))
    parser.add_argument("--folds", type=int, default=config.get("folds", 3))
    parser.add_argument("--clip-grid", nargs="+", default=config.get("clip_grid", [1, 5, 10, 20, 50, 100, 500, 1000, None]))
    parser.add_argument("--quantile-grid", type=float, nargs="*", default=config.get("quantile_grid", [.9, .95, .99, .995, .999]))
    parser.add_argument("--seed", type=int, default=config.get("seed", 42))
    parser.add_argument("--reference-bootstrap", type=int, default=config.get("reference_bootstrap", 1000))
    parser.add_argument("--metric-bootstrap", type=int, default=config.get("metric_bootstrap", 2000))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.clip_grid = [None if value is None or str(value).lower() == "none" else float(value) for value in args.clip_grid]
    except ValueError:
        parser.error("--clip-grid requires positive numbers or none")
    if any(value is not None and (not np.isfinite(value) or value <= 0) for value in args.clip_grid):
        parser.error("Clip bounds must be finite and positive")
    if len(set(args.clip_grid)) != len(args.clip_grid) or None not in args.clip_grid:
        parser.error("Clip grid must be unique and include none for paired comparisons")
    if (len(set(args.quantile_grid)) != len(args.quantile_grid)
            or any(not np.isfinite(q) or not 0 < q <= 1 for q in args.quantile_grid)):
        parser.error("Quantiles must be unique finite numbers in (0, 1]")
    if (args.folds < 2 or args.repeats < 2 or args.seed < 0
            or args.reference_bootstrap < 2 or args.metric_bootstrap < 2
            or any(n < args.folds for n in args.sample_sizes)
            or len(set(args.sample_sizes)) != len(args.sample_sizes)):
        parser.error("Invalid folds, sizes, repeats, seed, or bootstrap count")
    args.campaign = sorted(set(args.campaign))
    args.sample_sizes.sort()
    args.clip_grid = sorted((cap for cap in args.clip_grid if cap is not None)) + [None]
    args.quantile_grid.sort()
    return args


def summarize_tables(records, references, ranks, bootstrap_count, seed):
    frame = pd.DataFrame(records)
    keys = ["run_id", "campaign", "source", "window_id", "policy", "estimator", "n", "clip_id"]
    if frame.duplicated(keys + ["repeat"]).any():
        raise ValueError("Duplicate replicate primary keys")
    ref_lookup = {(row["campaign"], row["policy"]): row for row in references}
    summaries, ranking = [], []
    for key, group in frame.groupby(keys, sort=True):
        common = dict(zip(keys, key))
        reference = ref_lookup[(common["campaign"], common["policy"])]
        stats = summarize_estimates(group["estimate"], reference["value"])
        group_seed = int(identity(common)[:8], 16) ^ seed
        stats.update(bootstrap_metrics(group["estimate"], reference["value"], bootstrap_count, group_seed))
        baseline = frame
        for column in keys[:-1]:
            baseline = baseline[baseline[column] == common[column]]
        baseline = baseline[baseline.clip_id == "none"].set_index("repeat")["squared_error"]
        differences = group.set_index("repeat")["squared_error"] - baseline
        valid = differences.dropna().to_numpy()
        paired_ci = bootstrap_metrics(valid, 0, bootstrap_count, group_seed)
        stats.update(paired_valid_repeats=len(valid),
                     paired_mse_difference=float(valid.mean()) if len(valid) else np.nan,
                     paired_difference_ci_low=paired_ci["bias_ref_ci_low"],
                     paired_difference_ci_high=paired_ci["bias_ref_ci_high"])
        summaries.append(dict(common, reference=reference["value"],
                              reference_status=reference["reference_status"], **stats))
    rank_keys = ["run_id", "campaign", "source", "window_id", "n", "clip_id", "estimator"]
    for key, group in frame.groupby(rank_keys, sort=True):
        common = dict(zip(rank_keys, key))
        ranking.append(dict(common, **summarize_rankings(group, ranks[common["campaign"]])))
    return summaries, ranking


def run(args):
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with writer_lock(out):
        manifest_path = out / "manifest.json"
        previous = None
        if manifest_path.exists():
            if not args.resume:
                raise FileExistsError(f"{out} exists; use --resume or a fresh output directory")
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        versions = {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")}
        sources = sorted((ROOT / "ope" / "experiments").glob("*.py"))
        sources = [path for path in sources if path.name != "plotting.py"]
        sources.extend([ROOT / "ope" / "estimators.py", ROOT / "ope" / "reward_model.py", Path(__file__)])
        computation = dict(
            schema_version=SCHEMA_VERSION, seed=args.seed, folds=args.folds, window=args.window,
            clip_grid=args.clip_grid, quantile_grid=args.quantile_grid, versions=versions,
            python=platform.python_version(), blas_threads=1,
            reference_bootstrap=args.reference_bootstrap, metric_bootstrap=args.metric_bootstrap,
            sources={str(path.relative_to(ROOT)): file_digest(path) for path in sources})
        signature = identity(computation)
        if previous and previous["computation_signature"] != signature:
            raise ValueError("Resume computation settings, source code, or environment differ")
        run_id = previous["run_id"] if previous else identity(dict(signature=signature, output=str(out.resolve())))[:16]
        manifest = dict(schema_version=SCHEMA_VERSION, run_id=run_id,
                        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        computation_signature=signature, computation=computation,
                        requested=dict(campaigns=args.campaign, sample_sizes=args.sample_sizes, repeats=args.repeats),
                        sampling="independent uniform subsets; within-repeat without replacement",
                        interval_scope="conditional on fixed logs and fixed references; finite estimates",
                        provenance=dict(previous["provenance"]) if previous else {}, status="running")
        checkpoints = out / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        all_estimates, all_diagnostics, all_references, ranks = [], [], [], {}
        write_json(manifest_path, manifest)
        for campaign in args.campaign:
            print(f"{campaign}: validating inputs and recomputing Random reference", flush=True)
            started = time.perf_counter()
            inputs = load_inputs(campaign, args.processed_dir, args.policy_dir)
            rows, window = select_test_rows(inputs, args.window)
            if max(args.sample_sizes) >= len(rows["bts"]):
                raise ValueError(f"{campaign}: sample size must be smaller than {len(rows['bts'])} eligible BTS rows")
            input_signature = identity(inputs["fingerprints"])
            old_provenance = previous["provenance"].get(campaign) if previous else None
            if old_provenance and old_provenance["input_signature"] != input_signature:
                raise ValueError(f"{campaign}: resume input files differ")
            snapshot_directory = out / "inputs"
            snapshot_directory.mkdir(exist_ok=True)
            write_json(snapshot_directory / f"{campaign}_policy_manifest.json", inputs["export_manifest"])
            write_json(snapshot_directory / f"{campaign}_data_meta.json", inputs["meta"])
            manifest["provenance"][campaign] = dict(
                input_signature=input_signature, fingerprints=inputs["fingerprints"],
                window=window, source=inputs["meta"]["source"], status=inputs["provenance_status"])
            reference_seed = args.seed + ("all", "men", "women").index(campaign) * 10
            references, rank = compute_references(inputs, rows["random"],
                                                  bootstrap_repeats=args.reference_bootstrap, seed=reference_seed)
            for reference in references:
                reference.update(run_id=run_id, window_id=window["window_id"])
            all_references.extend(references)
            ranks[campaign] = rank
            reference_values = {row["policy"]: row["value"] for row in references}
            manifest["provenance"][campaign]["reference_rank"] = rank
            manifest["provenance"][campaign]["preflight_seconds"] = time.perf_counter() - started
            write_json(manifest_path, manifest)
            for n in args.sample_sizes:
                for repeat in range(args.repeats):
                    checkpoint_path = checkpoints / f"{campaign}_{n}_{repeat:04d}.json"
                    unit = dict(computation_signature=signature, input_signature=input_signature,
                                campaign=campaign, n=n, repeat=repeat, window_id=window["window_id"])
                    if checkpoint_path.exists():
                        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                        if saved["unit"] != unit or saved["digest"] != identity(saved["payload"]):
                            raise ValueError(f"Invalid checkpoint {checkpoint_path}")
                        estimates, diagnostics = saved["payload"]
                    else:
                        estimates, diagnostics = run_replicate(
                            inputs, rows["bts"], n=n, repeat=repeat, master_seed=args.seed,
                            folds=args.folds, clip_grid=args.clip_grid, quantile_grid=args.quantile_grid)
                        for record in estimates:
                            value = record["estimate"]
                            reference = reference_values[record["policy"]]
                            record.update(run_id=run_id, window_id=window["window_id"],
                                          reference=reference, squared_error=(value - reference) ** 2)
                        for record in diagnostics:
                            record.update(run_id=run_id, window_id=window["window_id"])
                        payload = clean_json([estimates, diagnostics])
                        write_json(checkpoint_path, dict(unit=unit, digest=identity(payload), payload=payload))
                    expected = len(POLICIES) * (len(args.clip_grid) + len(args.quantile_grid))
                    if len(estimates) != expected * 4 or len(diagnostics) != expected:
                        raise ValueError(f"Incomplete checkpoint {checkpoint_path}")
                    all_estimates.extend(estimates)
                    all_diagnostics.extend(diagnostics)
                    if (repeat + 1) % 10 == 0 or repeat + 1 == args.repeats:
                        print(f"{campaign}: n={n:,}, {repeat + 1}/{args.repeats} repeats ready", flush=True)
            del inputs
        summaries, ranking = summarize_tables(all_estimates, all_references, ranks, args.metric_bootstrap, args.seed)
        for filename, records in (
            ("references", all_references), ("replicates", all_estimates),
            ("diagnostics", all_diagnostics), ("summary", summaries), ("ranking", ranking),
        ):
            write_csv(out / f"{filename}.csv", records)
        manifest.update(status="complete", record_count=len(all_estimates),
                        summary_count=len(summaries), ranking_count=len(ranking),
                        output_sha256={f"{name}.csv": file_digest(out / f"{name}.csv") for name in
                                       ("references", "replicates", "diagnostics", "summary", "ranking")})
        write_json(manifest_path, manifest)
        print(f"Completed {out}: {len(all_estimates):,} estimates; {len(summaries):,} summary rows", flush=True)


def main():
    args = parse_args()
    with threadpool_limits(limits=1):
        run(args)


if __name__ == "__main__":
    main()
