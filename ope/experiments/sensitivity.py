"""Paired repeat-level experiments, descriptive error statistics, and ranking."""
from __future__ import annotations

import hashlib
import time
import warnings

import numpy as np

from ope.estimators import evaluate
from ope.reward_model import cross_fit
from .inputs import POLICIES, logged_weights

CAMPAIGN_CODES = {"all": 0, "men": 1, "women": 2}


def number_id(value):
    """Round-trip float spelling prevents distinct thresholds sharing an ID."""
    return str(float(value)).removesuffix(".0")


def resolve_thresholds(weights, clip_grid, quantile_grid):
    thresholds = []
    for cap in clip_grid:
        thresholds.append(dict(clip_id="none" if cap is None else f"cap_{number_id(cap)}",
                               clip_mode="none" if cap is None else "cap",
                               clip_value=cap, quantile=None, threshold_status="ok"))
    positive = weights[weights > 0]
    for quantile in quantile_grid:
        value = float(np.quantile(positive, quantile, method="linear")) if positive.size else None
        thresholds.append(dict(clip_id=f"qpos_{number_id(quantile)}", clip_mode="quantile",
                               clip_value=value, quantile=quantile,
                               threshold_status="ok" if positive.size else "zero_mass"))
    return thresholds


def weight_diagnostics(weights, cap):
    clipped = weights if cap is None else np.minimum(weights, cap)
    raw_total, total = weights.sum(), clipped.sum()
    raw_ess = raw_total ** 2 / np.square(weights).sum() if raw_total else 0.
    ess = total ** 2 / np.square(clipped).sum() if total else 0.
    return dict(mean_weight=float(clipped.mean()), max_weight=float(clipped.max()),
                raw_mean_weight=float(weights.mean()), raw_max_weight=float(weights.max()),
                positive_weight_fraction=float(np.mean(weights > 0)),
                clip_fraction=float(np.mean(weights > clipped)),
                removed_weight_fraction=float((raw_total - total) / raw_total) if raw_total else np.nan,
                ess_raw=float(raw_ess), ess_clipped=float(ess), ess_fraction=float(ess / len(weights)),
                weight_status="ok" if raw_total else "zero_mass")


def run_replicate(inputs, bts_rows, *, n, repeat, master_seed, folds, clip_grid, quantile_grid):
    campaign = inputs["campaign"]
    code = CAMPAIGN_CODES[campaign]
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, code, n, repeat, 0]))
    fit_seed = int(np.random.SeedSequence([master_seed, code, n, repeat, 1]).generate_state(1)[0])
    indices = rng.choice(bts_rows, size=n, replace=False)
    log = inputs["logs"]["bts"]
    sample = {key: log[key][indices] for key in ("context", "action", "position", "reward", "pscore")}
    sample.update(n_rounds=n, n_actions=log["n_actions"], len_list=log["len_list"])
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        q_hat = cross_fit(sample, folds=folds, seed=fit_seed)
    fit_seconds = time.perf_counter() - start
    common = dict(campaign=campaign, source=inputs["meta"]["source"], n=n, repeat=repeat,
                  sample_digest=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                  fit_seed=fit_seed)
    estimates, diagnostics = [], []
    for policy in POLICIES:
        distribution = inputs["distributions"]["bts"][policy][indices]
        weights = logged_weights(sample, distribution)
        for threshold in resolve_thresholds(weights, clip_grid, quantile_grid):
            base = dict(common, policy=policy, **threshold)
            values = evaluate(sample, distribution, q_hat, clip=threshold["clip_value"], allow_zero_mass=True)
            for estimator, value in values.items():
                estimates.append(dict(base, estimator=estimator, estimate=float(value),
                                      status="ok" if np.isfinite(value) else "undefined_zero_mass"))
            diagnostics.append(dict(
                base, sample_fraction=n / len(bts_rows), n_clicks=int(sample["reward"].sum()),
                fit_seconds=fit_seconds, fit_warning_count=len(caught),
                fit_warnings=" | ".join(sorted({str(item.message) for item in caught})),
                **weight_diagnostics(weights, threshold["clip_value"])))
    return estimates, diagnostics


def summarize_estimates(estimates, reference):
    values = np.asarray(estimates, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(reference):
        raise ValueError("Expected nonempty estimates and a finite reference")
    valid = values[np.isfinite(values)]
    result = dict(repeats=len(values), valid_repeats=len(valid), invalid_rate=1 - len(valid) / len(values))
    fields = ("mean_estimate", "bias_ref", "variance_emp", "mse_ref", "rmse_ref", "mean_mc_se", "mse_mc_se")
    result.update({key: np.nan for key in fields})
    if not valid.size:
        return result
    mean = float(valid.mean())
    errors = (valid - reference) ** 2
    result.update(mean_estimate=mean, bias_ref=mean - reference,
                  variance_emp=float(np.mean((valid - mean) ** 2)),
                  mse_ref=float(errors.mean()), rmse_ref=float(np.sqrt(errors.mean())))
    if len(valid) >= 2:
        result.update(mean_mc_se=float(valid.std(ddof=1) / np.sqrt(len(valid))),
                      mse_mc_se=float(errors.std(ddof=1) / np.sqrt(len(valid))))
    if not np.isclose(result["mse_ref"], result["bias_ref"] ** 2 + result["variance_emp"], rtol=1e-10, atol=1e-14):
        raise ArithmeticError("Empirical MSE decomposition failed")
    return result


def bootstrap_metrics(values, reference, repeats, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    names = ("bias_ref", "variance_emp", "mse_ref")
    result = {f"{name}_ci_{side}": np.nan for name in names for side in ("low", "high")}
    if len(values) < 2:
        return result
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(repeats, len(values)))]
    samples = (draws.mean(axis=1) - reference, draws.var(axis=1), ((draws - reference) ** 2).mean(axis=1))
    for name, sample in zip(names, samples):
        result[f"{name}_ci_low"], result[f"{name}_ci_high"] = np.quantile(sample, [.025, .975])
    return result


def summarize_rankings(records, reference_rank, tolerance=1e-12):
    table = records.pivot(index="repeat", columns="policy", values="estimate")
    valid = table.dropna(subset=list(POLICIES))
    difference = valid["linucb"] - valid["logisticts"]
    ties = np.abs(difference) <= tolerance
    ref = reference_rank["value"]
    ref_sign = 0 if abs(ref) <= tolerance else np.sign(ref)
    signs = np.sign(difference)
    signs[ties] = 0
    agreement = float(np.mean(signs == ref_sign)) if len(valid) else np.nan
    strict = signs[~ties]
    spearman = float(np.mean(strict * ref_sign)) if len(strict) and ref_sign else np.nan
    return dict(paired_valid_repeats=len(valid), repeats=len(table),
                tie_rate=float(ties.mean()) if len(valid) else np.nan,
                agreement_rate=agreement, spearman_mean=spearman,
                reference_difference=ref, reference_ci_low=reference_rank["ci_low"],
                reference_ci_high=reference_rank["ci_high"],
                reference_rank_status=reference_rank["status"], tie_tolerance=tolerance)
