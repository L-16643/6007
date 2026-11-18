import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ope.data import load_bandit_feedback
from ope.data.ground_truth import policy_value_from_random_logs
from ope.estimators import evaluate
from ope.experiments.inputs import (
    compute_references, load_inputs, row_digest, select_test_rows, timestamp_ns,
)
from ope.experiments.sensitivity import (
    resolve_thresholds, run_replicate, summarize_estimates, summarize_rankings, weight_diagnostics,
)
from scripts.run_sensitivity import parse_args, run, writer_lock
from ope.experiments.plotting import plot_clipping, plot_rankings, plot_run


@pytest.fixture
def artifacts(tmp_path):
    processed = tmp_path / "processed"
    policies = tmp_path / "policies"
    base = processed / "all"
    directory = policies / "all"
    directory.mkdir(parents=True)
    splits = {}
    rng = np.random.default_rng(3)
    for log, split, count, start in (
        ("random", "train", 100, 0),
        ("random", "test", 120, 100),
        ("bts", "test", 150, 90),
    ):
        target = base / f"{log}_{split}"
        target.mkdir(parents=True)
        times = np.arange(start, start + count, dtype=np.int64) + pd.Timestamp("2019-11-24", tz="UTC").value
        arrays = dict(timestamp=times, context=rng.normal(size=(count, 2)).astype("float32"),
                      action=rng.integers(2, size=count), position=rng.integers(2, size=count),
                      reward=rng.binomial(1, .15, size=count), pscore=np.full(count, .5))
        for key, array in arrays.items():
            np.save(target / f"{key}.npy", array)
        splits.setdefault(log, {})[split] = dict(
            n_rounds=count, time_start=str(pd.Timestamp(times[0], tz="UTC")),
            time_end=str(pd.Timestamp(times[-1], tz="UTC")))
    np.save(base / "action_context.npy", np.eye(2))
    meta = dict(campaign="all", source="sample", n_actions=2, len_list=2,
                context_feature_names=["x", "y"], splits=splits)
    (base / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    manifest = dict(campaign="all", source="sample", data_splits=splits,
                    context_feature_names=["x", "y"], test={})
    for log in ("random", "bts"):
        feedback = load_bandit_feedback("all", log, "test", processed)
        n = feedback["n_rounds"]
        pi = np.zeros((n, 2, 2), dtype="float32")
        pi[:, 0] = 1
        files = {}
        for name, dist in (("linucb", pi), ("logisticts", np.full_like(pi, .5))):
            files[name] = f"{name}_{log}_test.npy"
            np.save(directory / files[name], dist)
        manifest["test"][log] = dict(shape=list(pi.shape), files=files,
                                     row_keys_sha256=row_digest(feedback))
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return processed, policies


def test_metrics_missing_values_and_signed_errors():
    result = summarize_estimates([.1, .2, .3, np.nan], .15)
    assert result["valid_repeats"] == 3 and result["invalid_rate"] == .25
    assert result["bias_ref"] == pytest.approx(.05)
    assert result["variance_emp"] == pytest.approx(1 / 150)
    assert result["mse_ref"] == pytest.approx(11 / 1200)
    assert summarize_estimates([np.nan], 0)["valid_repeats"] == 0
    assert np.isnan(summarize_estimates([-.2], 0)["mean_mc_se"])
    with pytest.raises(ValueError):
        summarize_estimates([], 0)


def test_clipping_arithmetic_zero_mass_and_quantiles():
    log = dict(action=np.array([0, 1]), position=np.array([1, 0]), reward=np.array([1, 0]),
               pscore=np.array([.5, .5]), n_actions=2, len_list=2)
    pi = np.array([[[.5, 1], [.5, 0]], [[.75, 0], [.25, 1]]])
    q = np.full((2, 2, 2), .4)
    assert evaluate(log, pi, q, clip=.75) == pytest.approx(dict(DM=.4, IPS=.375, SNIPS=.6, DR=.525))
    assert evaluate(log, pi, q, clip=10) == pytest.approx(evaluate(log, pi, q))
    thresholds = resolve_thresholds(np.array([0., 0., 2., 4.]), [None], [.5])
    assert thresholds[-1]["clip_value"] == 3.
    assert resolve_thresholds(np.zeros(2), [None], [.9])[-1]["threshold_status"] == "zero_mass"
    assert weight_diagnostics(np.zeros(2), None)["ess_raw"] == 0
    thresholds = resolve_thresholds(np.ones(2), [1., 1.00000001, None], [])
    assert len({row["clip_id"] for row in thresholds}) == 3


def test_timestamp_units_and_reference_recalculation(artifacts):
    processed, policies = artifacts
    inputs = load_inputs("all", processed, policies)
    rows, bounds = select_test_rows(inputs, "intersection")
    assert len(rows["bts"]) == len(rows["random"]) == 120
    assert rows["bts"][0] == 10
    references, _ = compute_references(inputs, rows["random"], bootstrap_repeats=10, seed=5)
    feedback = inputs["logs"]["random"]
    expected = policy_value_from_random_logs(feedback, inputs["distributions"]["random"]["linucb"], n_bootstrap=10, seed=5)
    assert references[0]["value"] == expected["value"]
    assert references[0]["se"] == expected["se"]
    assert references[0]["bootstrap_ci_low"] == expected["bootstrap_ci_low"]
    for factor in (1, 1000, 1_000_000, 1_000_000_000):
        times = np.array([1_600_000_000, 1_600_000_001], dtype=np.int64) * (1_000_000_000 // factor)
        split = dict(time_start="2020-09-13 12:26:40+00:00", time_end="2020-09-13 12:26:41+00:00")
        assert np.array_equal(timestamp_ns(times, split), np.array([1_600_000_000, 1_600_000_001]) * 1_000_000_000)
    with pytest.raises(ValueError):
        timestamp_ns(np.array([2, 1]), split)
    assert bounds["mode"] == "intersection"
    feedback["reward"][:] = 0
    changed, _ = compute_references(inputs, rows["random"], bootstrap_repeats=10, seed=5)
    assert changed[0]["reference_status"] == "zero_contributions"


def test_row_corruption_rejected(artifacts):
    processed, policies = artifacts
    path = processed / "all" / "bts_test" / "action.npy"
    array = np.load(path)
    np.save(path, 1 - array)
    with pytest.raises(ValueError, match="row order"):
        load_inputs("all", processed, policies)


def test_replicate_shares_fit_and_is_order_independent(artifacts):
    processed, policies = artifacts
    inputs = load_inputs("all", processed, policies)
    rows, _ = select_test_rows(inputs, "intersection")
    kwargs = dict(n=40, repeat=2, master_seed=42, folds=3, quantile_grid=[.9])
    q = np.full((40, 2, 2), .1)
    with patch("ope.experiments.sensitivity.cross_fit", return_value=q) as fit:
        first, _ = run_replicate(inputs, rows["bts"], clip_grid=[.5, 2, None], **kwargs)
        assert fit.call_count == 1
    with patch("ope.experiments.sensitivity.cross_fit", return_value=q):
        second, _ = run_replicate(inputs, rows["bts"], clip_grid=[None, 2, .5], **kwargs)
    columns = ["policy", "clip_id", "estimator"]
    left = pd.DataFrame(first).set_index(columns).sort_index()
    right = pd.DataFrame(second).set_index(columns).sort_index()
    pd.testing.assert_frame_equal(left, right)
    assert left.reset_index().query("estimator == 'DM'").groupby("policy").estimate.nunique().eq(1).all()


def test_two_policy_ranking_ties_and_missing():
    frame = pd.DataFrame([
        dict(repeat=r, policy=p, estimate=value)
        for r, pair in enumerate(((.2, .1), (.1, .2), (.1, .1), (np.nan, .1)))
        for p, value in zip(("linucb", "logisticts"), pair)
    ])
    result = summarize_rankings(frame, dict(value=.1, ci_low=-.1, ci_high=.2, status="uncertain"))
    assert result["paired_valid_repeats"] == 3
    assert result["tie_rate"] == pytest.approx(1 / 3)
    assert result["agreement_rate"] == pytest.approx(1 / 3)
    assert result["spearman_mean"] == 0


def test_cli_resume_extension_and_tamper_detection(artifacts, tmp_path):
    processed, policies = artifacts
    output = tmp_path / "run"
    argv = ["--campaign", "all", "--processed-dir", str(processed), "--policy-dir", str(policies),
            "--out-dir", str(output), "--sample-sizes", "30", "--repeats", "2",
            "--clip-grid", "2", "none", "--quantile-grid", ".9",
            "--reference-bootstrap", "10", "--metric-bootstrap", "10"]
    run(parse_args(argv))
    original = pd.read_csv(output / "replicates.csv")
    assert len(original) == 48
    figures = tmp_path / "figures"
    plot_run(output, figures, tmp_path / "poster", tmp_path / "report.md")
    assert (figures / "clipping_mse_all.pdf").stat().st_size > 1000
    assert (tmp_path / "poster" / "generated_results.tex").exists()
    run(parse_args(argv + ["--resume"]))
    pd.testing.assert_frame_equal(original, pd.read_csv(output / "replicates.csv"))
    # An incomplete run retains its complete units and recomputes the missing one.
    (output / "checkpoints" / "all_30_0001.json").unlink()
    run(parse_args(argv + ["--resume"]))
    pd.testing.assert_frame_equal(original, pd.read_csv(output / "replicates.csv"))
    expanded = argv.copy()
    expanded[expanded.index("--repeats") + 1] = "3"
    run(parse_args(expanded + ["--resume"]))
    assert len(pd.read_csv(output / "replicates.csv")) == 72
    checkpoint = output / "checkpoints" / "all_30_0000.json"
    saved = json.loads(checkpoint.read_text())
    saved["payload"][0][0]["estimate"] += 1
    checkpoint.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="checkpoint"):
        run(parse_args(expanded + ["--resume"]))


@pytest.mark.parametrize("option,value", [("--clip-grid", "nan"), ("--quantile-grid", "0"),
                                         ("--folds", "1"), ("--seed", "-1")])
def test_invalid_cli_arguments(option, value):
    with pytest.raises(SystemExit):
        parse_args([option, value])


def test_single_writer_lock(tmp_path):
    with writer_lock(tmp_path):
        with pytest.raises(OSError):
            with writer_lock(tmp_path):
                pytest.fail("Second writer acquired an occupied run")


def test_clipping_axes_include_all_estimators_and_intervals(tmp_path):
    import matplotlib.pyplot as plt
    records = []
    for policy in ("linucb", "logisticts"):
        for estimator, value in (("DM", 1e-6), ("IPS", 1e-4), ("SNIPS", 1e-3), ("DR", .01)):
            for clip_id in ("cap_1", "none"):
                records.append(dict(campaign="all", n=20, policy=policy, estimator=estimator,
                                    clip_id=clip_id, mse_ref=value, mse_ref_ci_low=0 if estimator == "DM" else value / 2,
                                    mse_ref_ci_high=value * 1.5, variance_emp=value / 2,
                                    bias_ref=np.sqrt(value / 2)))
    captured = {}

    def capture(figure, out, stem, caption, files):
        captured[stem] = [axis.get_ylim() for axis in figure.axes]
        plt.close(figure)

    with patch("ope.experiments.plotting.save_figure", side_effect=capture):
        plot_clipping(pd.DataFrame(records), "all", 20, tmp_path, "test", {})
    assert all(low == 0 and high >= .015 for low, high in captured["clipping_mse_all"])


@pytest.mark.parametrize("tie_rate", [0.0, 0.25])
def test_ranking_panels_share_one_legend_outside_all_titles(tmp_path, tie_rate):
    import matplotlib.pyplot as plt
    campaigns = ["all", "men", "women"]
    records = [dict(campaign=campaign, n=20, estimator=estimator, clip_id=rule,
                    agreement_rate=.75, tie_rate=tie_rate, paired_valid_repeats=4,
                    reference_rank_status="uncertain")
               for campaign in campaigns for estimator in ("DM", "IPS", "SNIPS", "DR")
               for rule in ("cap_1", "none")]

    def check(figure, out, stem, caption, files):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        assert len(figure.axes) == len(campaigns) * (2 if tie_rate else 1)
        assert all(axis.get_legend() is None for axis in figure.axes)
        assert len(figure.legends) == 1
        legend = figure.legends[0].get_window_extent(renderer)
        assert figure.bbox.contains(*legend.p0) and figure.bbox.contains(*legend.p1)
        assert legend.y0 > max(axis.get_tightbbox(renderer).y1 for axis in figure.axes)
        assert ("No estimated ties." in caption) == (tie_rate == 0)
        plt.close(figure)

    with patch("ope.experiments.plotting.save_figure", side_effect=check):
        plot_rankings(pd.DataFrame(records), campaigns, 20, tmp_path, "test", {})


def test_symlog_axes_keep_curves_and_tick_labels_inside_frame(tmp_path):
    """Headroom regression: peaks must stay off the top spine and no visible
    y tick label may extend above the axes frame."""
    import matplotlib.pyplot as plt
    records = []
    for policy in ("linucb", "logisticts"):
        for estimator, value in (("DM", 1e-6), ("IPS", 1e-4), ("SNIPS", 1e-3), ("DR", .01)):
            records.append(dict(campaign="all", n=20, policy=policy, estimator=estimator,
                                clip_id="none", mse_ref=value, mse_ref_ci_low=value / 2,
                                mse_ref_ci_high=value * 1.5, variance_emp=value / 2,
                                bias_ref=np.sqrt(value / 2)))
    figures = {}

    def capture(figure, out, stem, caption, files):
        figures[stem] = figure
        plt.close(figure)

    with patch("ope.experiments.plotting.save_figure", side_effect=capture):
        plot_clipping(pd.DataFrame(records), "all", 20, tmp_path, "test", {})
    for figure in figures.values():
        figure.canvas.draw()
        inv = figure.transFigure.inverted()
        for axis in figure.axes:
            if axis.get_yscale() != "symlog":
                continue
            top = inv.transform(axis.get_window_extent().get_points())[1, 1]
            for label in axis.get_yticklabels():
                if not label.get_text():
                    continue
                extent = inv.transform(label.get_window_extent().get_points())
                assert extent[1, 1] <= top + 1e-9, \
                    f"tick label {label.get_text()} extends above the axes frame"
    plt.close("all")


def test_symlog_axes_raise_floor_for_narrow_data(tmp_path):
    """The y floor must rise to just below the data when the data spans fewer
    than three decades, so curves are not squeezed into the top of the axis."""
    import matplotlib.pyplot as plt
    records = []
    for policy in ("linucb", "logisticts"):
        for estimator, value in (("DM", 1e-5), ("IPS", 4e-5), ("SNIPS", 6e-5), ("DR", 1.5e-4)):
            records.append(dict(campaign="all", n=20, policy=policy, estimator=estimator,
                                clip_id="none", mse_ref=value, mse_ref_ci_low=value / 2,
                                mse_ref_ci_high=value * 1.5, variance_emp=value / 2,
                                bias_ref=np.sqrt(value / 2)))
    captured = {}

    def capture(figure, out, stem, caption, files):
        captured[stem] = figure
        plt.close(figure)

    with patch("ope.experiments.plotting.save_figure", side_effect=capture):
        plot_clipping(pd.DataFrame(records), "all", 20, tmp_path, "test", {})
    for figure in captured.values():
        for axis in figure.axes:
            if axis.get_yscale() == "symlog":
                assert axis.get_ylim()[0] > 0, "narrow data should raise the symlog floor"
    plt.close("all")
