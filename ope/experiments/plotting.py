"""Create traceable publication figures and a Chinese experimental report."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation, PercentFormatter, ScalarFormatter
import numpy as np
import pandas as pd

from .inputs import POLICIES, file_digest
from .sensitivity import number_id

COLORS = {"DM": "#0072B2", "IPS": "#D55E00", "SNIPS": "#009E73", "DR": "#9A60A3"}
MARKERS = {"DM": "o", "IPS": "s", "SNIPS": "^", "DR": "D"}
POLICY_LABELS = {"linucb": "LinUCB", "logisticts": "LogisticTS"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


def load_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["status"] != "complete":
        raise ValueError("Figures require a complete schema-1 run")
    tables = {}
    for filename, digest in manifest["output_sha256"].items():
        if file_digest(directory / filename) != digest:
            raise ValueError(f"Result checksum mismatch: {filename}")
        tables[Path(filename).stem] = pd.read_csv(
            directory / filename,
            dtype={"run_id": str, "window_id": str, "sample_digest": str},
        )
    summary = tables["summary"]
    if summary.run_id.nunique() != 1 or summary.run_id.iloc[0] != manifest["run_id"]:
        raise ValueError("Mixed result identities")
    if summary.duplicated(["campaign", "policy", "estimator", "n", "clip_id"]).any():
        raise ValueError("Duplicate summary keys")
    if not summary.repeats.eq(manifest["requested"]["repeats"]).all():
        raise ValueError("Incomplete repeat counts")
    if not np.allclose(summary.mse_ref, summary.bias_ref ** 2 + summary.variance_emp,
                       rtol=1e-10, atol=1e-14, equal_nan=True):
        raise ValueError("Error decomposition mismatch")
    return manifest, tables


def save_figure(figure, out, stem, caption, files):
    footer = .32 / figure.get_figheight()
    figure.get_layout_engine().set(rect=(0, footer, 1, 1 - footer))
    figure.text(.5, .08 / figure.get_figheight(), caption, fontsize=8, ha="center", va="bottom")
    # Resolve the constrained layout before switching to the PDF renderer.
    figure.canvas.draw()
    for suffix in ("pdf", "png"):
        path = out / f"{stem}.{suffix}"
        figure.savefig(path, bbox_inches="tight", facecolor="white")
        files[path.name] = file_digest(path)
    plt.close(figure)


def legend_above(figure, axis):
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncols=len(labels),
                  fontsize=8, handlelength=1.6, columnspacing=1.3, frameon=False)


def fixed_bounds(frame):
    caps = sorted(float(value[4:]) for value in frame.clip_id.unique() if value.startswith("cap_"))
    return ([f"cap_{number_id(cap)}" for cap in caps] + ["none"],
            [f"{cap:g}" for cap in caps] + ["None"])


def comparison_stem(metric, campaigns):
    suffix = campaigns[0] if len(campaigns) == 1 else "by_campaign"
    return f"{metric}_{suffix}"


def size_ticks(axis, sizes):
    axis.set_xscale("log")
    axis.set_xticks(sizes, [f"{n/1000:g}k" if n >= 1000 else str(n) for n in sizes])


def error_axis(axis, logarithmic=False):
    if logarithmic:
        axis.set_yscale("symlog", linthresh=1e-8)
        values = [v for line in axis.lines for v in np.asarray(line.get_ydata()).ravel()]
        for collection in axis.collections:
            for path in collection.get_paths():
                values.extend(path.vertices[:, 1])
        positive = [v for v in values if v > 0 and np.isfinite(v)]
        if positive:
            tick_floor = min(positive) / 2
            base = 0 if 0 in values else tick_floor
            transform = axis.yaxis.get_transform()
            lo, hi = transform.transform([base, max(positive)])
            new_top = float(transform.inverted().transform([lo + (hi - lo) / .90])[0])
            axis.set_ylim(base, new_top)
            decades = np.log10(new_top / tick_floor)
            steps = ((1, 2, 5) if decades <= 1.6 else
                     (1, 5) if decades <= 3 else
                     (1, 3) if decades <= 5 else (1,))
            ticks = [mult * 10.0 ** k
                     for k in range(int(np.floor(np.log10(tick_floor))) - 1,
                                    int(np.ceil(np.log10(new_top))) + 2)
                     for mult in steps
                     if tick_floor <= mult * 10.0 ** k <= max(positive)]
            axis.set_yticks(([0] if base == 0 else []) + ticks)
            axis.yaxis.set_major_formatter(LogFormatterSciNotation(
                base=10, minor_thresholds=(np.inf, np.inf), linthresh=1e-8))
        else:
            axis.set_ylim(0, 1e-8)
            axis.set_yticks([0])
    else:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        axis.yaxis.set_major_formatter(formatter)


def plot_clipping(summary, campaigns, n, out, caption, files):
    if isinstance(campaigns, str):
        campaigns = [campaigns]
    chosen = summary[summary.n == n]
    ids, labels = fixed_bounds(chosen)
    x = np.arange(len(ids))
    fig, axes = plt.subplots(len(campaigns), 2, figsize=(9, 2.35 * len(campaigns) + .75),
                             squeeze=False, layout="constrained")
    for row, campaign in enumerate(campaigns):
        for col, policy in enumerate(POLICIES):
            axis = axes[row, col]
            panel = chosen[(chosen.campaign == campaign) & (chosen.policy == policy)]
            for estimator, color in COLORS.items():
                data = panel[panel.estimator == estimator].set_index("clip_id").reindex(ids)
                axis.plot(x, data.mse_ref, marker=MARKERS[estimator], ms=3, lw=1.2,
                          label=estimator, color=color)
                axis.fill_between(x, data.mse_ref_ci_low, data.mse_ref_ci_high, color=color, alpha=.09)
            axis.set(title=f"{campaign.title()} / {POLICY_LABELS[policy]}",
                     xlabel="Importance-weight upper bound", ylabel="Reference MSE")
            axis.set_xticks(x, labels, rotation=35)
            error_axis(axis, True)
    legend_above(fig, axes[0, 0])
    save_figure(fig, out, comparison_stem("clipping_mse", campaigns),
                caption + "  Shading: conditional 95% repeat-bootstrap intervals.", files)


def plot_clipping_details(summary, campaign, n, out, caption, files):
    chosen = summary[(summary.campaign == campaign) & (summary.n == n)]
    ids, labels = fixed_bounds(chosen)
    x = np.arange(len(ids))
    decomposition, dec_axes = plt.subplots(2, 4, figsize=(11, 5.1), layout="constrained")
    for row, policy in enumerate(POLICIES):
        for col, estimator in enumerate(COLORS):
            data = chosen[(chosen.policy == policy) & (chosen.estimator == estimator)].set_index("clip_id").reindex(ids)
            dec = dec_axes[row, col]
            dec.plot(x, data.bias_ref ** 2, label="Bias(ref) squared", color="#0072B2", lw=1.1)
            dec.plot(x, data.variance_emp, label="Empirical variance", color="#D55E00", lw=1.1)
            dec.plot(x, data.mse_ref, label="Reference MSE", color="#333333", lw=1.1)
            dec.set(title=f"{POLICY_LABELS[policy]} / {estimator}", xlabel="Weight bound")
            dec.set_xticks(x, labels, rotation=50, fontsize=8)
            error_axis(dec, True)
    legend_above(decomposition, dec_axes[0, 0])
    save_figure(decomposition, out, f"clipping_decomposition_{campaign}", caption, files)
    quantile = chosen[chosen.clip_id.str.startswith("qpos_")]
    if not quantile.empty:
        figure, axes = plt.subplots(1, 2, figsize=(9, 3), layout="constrained")
        for axis, policy in zip(axes, POLICIES):
            for estimator, color in COLORS.items():
                data = quantile[(quantile.policy == policy) & (quantile.estimator == estimator)].copy()
                data["q"] = data.clip_id.str.removeprefix("qpos_").astype(float)
                data = data.sort_values("q")
                axis.plot(data.q, data.mse_ref, label=estimator, color=color, marker=MARKERS[estimator],
                          ms=2.5, lw=1.1)
            axis.set(title=POLICY_LABELS[policy], xlabel="Quantile of positive weights", ylabel="Reference MSE")
            error_axis(axis, True)
        legend_above(figure, axes[0])
        save_figure(figure, out, f"quantile_mse_{campaign}", caption, files)


def plot_sizes(summary, campaign, out, caption, files):
    chosen = summary[(summary.campaign == campaign) & (summary.clip_id == "none")]
    figure, axes = plt.subplots(2, 3, figsize=(10, 5.3), layout="constrained")
    for row, policy in enumerate(POLICIES):
        for col, (metric, label) in enumerate((("bias_ref", "Signed bias vs reference"),
                                               ("variance_emp", "Empirical variance"),
                                               ("mse_ref", "Reference MSE"))):
            axis = axes[row, col]
            for estimator, color in COLORS.items():
                data = chosen[(chosen.policy == policy) & (chosen.estimator == estimator)].sort_values("n")
                axis.plot(data.n, data[metric], label=estimator, color=color,
                          marker=MARKERS[estimator], ms=2.5, lw=1.1)
                axis.fill_between(data.n, data[f"{metric}_ci_low"],
                                  data[f"{metric}_ci_high"], color=color, alpha=.08)
            axis.set(title=f"{POLICY_LABELS[policy]} / {label}", xlabel="BTS subsample rows")
            size_ticks(axis, sorted(chosen.n.unique()))
            if metric == "bias_ref":
                axis.axhline(0, color="#555555", lw=.7)
            error_axis(axis, metric != "bias_ref")
    legend_above(figure, axes[0, 0])
    save_figure(figure, out, f"sample_size_{campaign}", caption + " Unclipped; shading: conditional 95% repeat-bootstrap intervals.", files)


def plot_variance(summary, campaigns, n, out, caption, files):
    chosen = summary[(summary.n == n) & (summary.clip_id == "none")]
    fig, axes = plt.subplots(len(campaigns), 2, figsize=(9, 1.7 * len(campaigns) + .85),
                             squeeze=False, layout="constrained")
    for row, campaign in enumerate(campaigns):
        for col, policy in enumerate(POLICIES):
            axis = axes[row, col]
            values = chosen[(chosen.campaign == campaign) & (chosen.policy == policy)]
            values = values.set_index("estimator").reindex(list(COLORS))
            axis.set_axisbelow(True)
            axis.grid(axis="x", visible=False)
            axis.bar(values.index, values.variance_emp, width=.55, label="Empirical variance",
                     color="#32739A")
            axis.bar(values.index, values.bias_ref ** 2, bottom=values.variance_emp, width=.55,
                     label="Squared bias vs reference", color="#F3CB77", edgecolor="#826632",
                     hatch="///", linewidth=.45)
            axis.set(title=f"{campaign.title()} / {POLICY_LABELS[policy]}", ylabel="Squared click rate")
            error_axis(axis)
    legend_above(fig, axes[0, 0])
    save_figure(fig, out, comparison_stem("variance_decomposition", campaigns),
                caption + "  Unclipped. Total bar height = reference MSE. Scales vary by panel.", files)


def plot_weight_diagnostics(diagnostics, campaigns, n, out, caption, files):
    fixed = diagnostics[(diagnostics.n == n) & ~diagnostics.clip_id.str.startswith("qpos_")]
    ids, labels = fixed_bounds(fixed)
    metrics = {"ess_fraction": "ESS / n", "clip_fraction": "Rows clipped",
               "removed_weight_fraction": "Weight removed"}
    figure, axes = plt.subplots(len(campaigns), 3, figsize=(10, 2 * len(campaigns) + .7),
                                squeeze=False, layout="constrained", sharey="col")
    means = fixed.groupby(["campaign", "policy", "clip_id"])[list(metrics)].mean()
    for row, campaign in enumerate(campaigns):
        for col, (key, label) in enumerate(metrics.items()):
            axis = axes[row, col]
            for policy, color, marker in zip(POLICIES, ("#0072B2", "#D55E00"), ("o", "s")):
                values = means.loc[(campaign, policy), key].reindex(ids)
                axis.plot(range(len(ids)), values, label=POLICY_LABELS[policy],
                          color=color, marker=marker, ms=3, lw=1.2)
            axis.set(title=f"{campaign.title()} / {label}", xlabel="Weight bound")
            axis.set_xticks(range(len(ids)), labels, rotation=35)
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    for axis in axes[0]:
        axis.set_ylim(bottom=0)
    legend_above(figure, axes[0, 0])
    save_figure(figure, out, comparison_stem("weight_diagnostics", campaigns),
                caption + "  Points are means across repeats.", files)


def plot_rankings(ranking, campaigns, n, out, caption, files):
    fixed = ranking[(ranking.n == n) & ~ranking.clip_id.str.startswith("qpos_")]
    ids, labels = fixed_bounds(fixed)
    metrics = {"agreement_rate": "Agreement with reference order"}
    if not fixed.tie_rate.eq(0).all():
        metrics["tie_rate"] = "Estimated ties"
    figure, axes = plt.subplots(len(metrics), len(campaigns), squeeze=False, layout="constrained",
                                figsize=(max(6.5, 3.4 * len(campaigns)), 2.4 * len(metrics) + .95))
    for col, campaign in enumerate(campaigns):
        panel = fixed[fixed.campaign == campaign]
        status = panel.reference_rank_status.iloc[0]
        for row, (key, label) in enumerate(metrics.items()):
            axis = axes[row, col]
            for estimator, color in COLORS.items():
                values = panel[panel.estimator == estimator].set_index("clip_id").reindex(ids)
                axis.plot(range(len(ids)), values[key], label=estimator, color=color,
                          marker=MARKERS[estimator], ms=3, lw=1.2)
            axis.set(title=f"{campaign.title()} / reference {status}", ylim=(-.035, 1.035),
                     ylabel=label, xlabel="Weight bound")
            axis.set_yticks([0, .25, .5, .75, 1])
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
            axis.set_xticks(range(len(ids)), labels, rotation=35)
    legend_above(figure, axes[0, 0])
    low, high = fixed.paired_valid_repeats.min(), fixed.paired_valid_repeats.max()
    counts = str(low) if low == high else f"{low}–{high}"
    note = f"  Valid pairs per point: {counts}. " + ("No estimated ties." if len(metrics) == 1 else "")
    save_figure(figure, out, comparison_stem("ranking_agreement", campaigns), caption + note.rstrip(), files)


def poster_assets(manifest, tables, poster_dir):
    poster_dir.mkdir(parents=True, exist_ok=True)
    campaign = "all" if "all" in manifest["requested"]["campaigns"] else manifest["requested"]["campaigns"][0]
    maximum = max(manifest["requested"]["sample_sizes"])
    chosen = tables["summary"]
    chosen = chosen[(chosen.campaign == campaign) & (chosen.n == maximum)]
    fixed = chosen[~chosen.clip_id.str.startswith("qpos_")]
    ids, labels = fixed_bounds(fixed)
    figure, axes = plt.subplots(2, 1, figsize=(3.35, 2.7), layout="constrained", sharex=True)
    for axis, policy in zip(axes, POLICIES):
        for estimator, color in COLORS.items():
            values = fixed[(fixed.policy == policy) & (fixed.estimator == estimator)].set_index("clip_id").reindex(ids)
            axis.plot(range(len(ids)), values.mse_ref, color=color, label=estimator,
                      marker=MARKERS[estimator], ms=2.5, lw=1)
        axis.set_ylabel("Ref. MSE", fontsize=8)
        axis.set_title(POLICY_LABELS[policy], fontsize=8, pad=2)
        axis.tick_params(labelsize=7)
        error_axis(axis, True)
    legend_above(figure, axes[0])
    axes[1].set_xticks(range(len(ids)), labels, rotation=50, fontsize=8)
    axes[1].set_xlabel("Importance-weight upper bound", fontsize=8)
    figure.canvas.draw()
    figure.savefig(poster_dir / "clipping_mse.pdf", bbox_inches="tight")
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(3.35, 1.9), layout="constrained")
    for estimator, color in COLORS.items():
        values = tables["summary"]
        values = values[(values.campaign == campaign) & (values.policy == "logisticts")
                        & (values.estimator == estimator) & (values.clip_id == "none")].sort_values("n")
        axis.plot(values.n, values.mse_ref, color=color, label=estimator,
                  marker=MARKERS[estimator], ms=2.5, lw=1)
    axis.set(xlabel="BTS subsample size", ylabel="Reference MSE")
    axis.tick_params(labelsize=7)
    axis.xaxis.label.set_size(8)
    axis.yaxis.label.set_size(8)
    size_ticks(axis, sorted(tables["summary"].n.unique()))
    error_axis(axis, True)
    legend_above(figure, axis)
    figure.canvas.draw()
    figure.savefig(poster_dir / "sample_size.pdf", bbox_inches="tight")
    plt.close(figure)
    lines = [f"% Generated from run {manifest['run_id']}; raw reward units.",
             rf"\newcommand{{\RunID}}{{{manifest['run_id']}}}",
             rf"\newcommand{{\MaxSample}}{{{maximum:,}}}",
             rf"\newcommand{{\RepeatCount}}{{{manifest['requested']['repeats']}}}",
             rf"\newcommand{{\RunCampaign}}{{{campaign.title()}}}"]
    rows = []
    for policy in POLICIES:
        data = chosen[(chosen.policy == policy) & (chosen.estimator == "DR") & ~chosen.clip_id.str.startswith("qpos_")]
        best = data.loc[data.mse_ref.idxmin()]
        baseline = data[data.clip_id == "none"].iloc[0]
        reduction = 100 * (1 - best.mse_ref / baseline.mse_ref) if baseline.mse_ref else 0
        rows.append(f"{policy} & {best.clip_id.replace('cap_', '')} & {baseline.mse_ref:.2e} & {best.mse_ref:.2e} \\\\")
        name = "Lin" if policy == "linucb" else "TS"
        lines.append(rf"\newcommand{{\{name}Reduction}}{{{reduction:.1f}}}")
        lines.append(rf"\newcommand{{\{name}Bound}}{{{best.clip_id.replace('cap_', '')}}}")
    lines.append(r"\newcommand{\ResultRows}{" + "\n".join(rows) + "}")
    ranks = manifest["provenance"][campaign]["reference_rank"]
    lines.append(r"\newcommand{\RankStatus}{" + ranks["status"] + "}")
    (poster_dir / "generated_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (poster_dir / "results_provenance.json").write_text(json.dumps({
        "run_id": manifest["run_id"], "campaign": campaign, "n": maximum,
        "source": manifest["provenance"][campaign]["source"],
        "selection": "largest-scale All campaign; maximum sample size; full fixed grid shown",
        "csv_sha256": manifest["output_sha256"], "tables": rows,
    }, indent=2) + "\n", encoding="utf-8")


def write_report(manifest, tables, path, run_dir, figure_dir):
    summary, refs = tables["summary"], tables["references"]
    maximum = max(manifest["requested"]["sample_sizes"])
    chosen = summary[(summary.n == maximum) & ~summary.clip_id.str.startswith("qpos_")]
    lines = [
        "# 成员 E 实验分析与交付报告", "",
        f"本报告来自运行 {manifest['run_id']}，Git 基准为 {manifest['git_commit']}。"
        f"使用 {manifest['requested']['repeats']} 次重复，样本量为 {manifest['requested']['sample_sizes']}。"
        f"目标策略由 Random/train 冻结训练，每个 BTS 子样本重新进行 {manifest['computation']['folds']} 折交叉拟合，全部策略和截断规则共享预测。", "",
        "## 统计定义", "",
        "对每条已展示动作与推荐位，权重为 w=πe(a|x,position)/pscore，截断后为 min(w,c)。"
        "IPS 使用截断权重乘点击奖励，SNIPS 再以权重总和归一化，DR 使用目标分布下的奖励预测加截断加权残差。"
        "无截断规则使用原始权重，DM 对全部阈值保持相同。", "",
        "相对固定参考值 g，经验偏差为 mean(V̂)−g，经验方差为 mean((V̂−mean(V̂))²)，"
        "参考 MSE 为 mean((V̂−g)²)。三者满足 MSE=bias²+variance。"
        "图表使用原始奖励单位和对应平方单位，结果保留负 DR 估计及无定义 SNIPS 状态。", "",
        "## 参考值与窗口", "",
        "Random 日志 IPS 估计作为固定参考。表中区间为按 Random 行重采样的 95% bootstrap 区间。"
        "共同测试时间窗口减少时间段差异，用户上下文及推荐位分布仍可能影响跨日志可比性。", "",
        "| Campaign | 策略 | 参考 CTR | 95% 区间 | ESS | 非零贡献数 |", "|---|---|---|---|---|---|",
    ]
    for row in refs.itertuples():
        lines.append(f"| {row.campaign} | {row.policy} | {row.value:.6%} | "
                     f"[{row.bootstrap_ci_low:.6%}, {row.bootstrap_ci_high:.6%}] | {row.ess:.0f} | {row.positive_contribution_count} |")
    for campaign in manifest["requested"]["campaigns"]:
        window = manifest["provenance"][campaign]["window"]
        text = "，".join(f"{name} 共 {info['n_rounds']:,} 行，从 {info['start_utc']} 到 {info['end_utc']}"
                         for name, info in window["logs"].items())
        lines.extend(["", f"{campaign} 使用 {window['mode']} 窗口，{text}。"])
    lines.extend(["", "## 固定上限截断", "",
                  f"下表使用每组最大样本量 {maximum:,}，列出 DR 在固定网格中的最低参考 MSE。"
                  "配对差为该规则的逐次平方误差减去无截断平方误差，区间在共同有效 repeat 上重采样。", "",
                  "| Campaign | 策略 | 观察最低点 | 无截断 MSE | 网格最低 MSE | 配对 MSE 差及 95% 区间 |",
                  "|---|---|---|---|---|---|"])
    for (campaign, policy), group in chosen[chosen.estimator == "DR"].groupby(["campaign", "policy"]):
        best = group.loc[group.mse_ref.idxmin()]
        baseline = group[group.clip_id == "none"].iloc[0]
        lines.append(f"| {campaign} | {policy} | {best.clip_id} | {baseline.mse_ref:.5g} | {best.mse_ref:.5g} | "
                     f"{best.paired_mse_difference:.5g} [{best.paired_difference_ci_low:.5g}, {best.paired_difference_ci_high:.5g}] |")
    lines.extend(["", "这些最低点由同一网格的结果选出，后续泛化效果需要独立数据评估。"
                  "对每个阈值的原始结果、MC 标准误及有效次数均保存在 summary.csv。", "",
                  "## 无截断估计器比较", "",
                  f"以下数据均采用 n={maximum:,}。相对参考值的经验偏差保留正负号。", "",
                  "| Campaign | 策略 | 估计器 | Bias(ref) | 经验方差 | 参考 MSE | 有效重复数 |",
                  "|---|---|---|---|---|---|---|"])
    for row in chosen[chosen.clip_id == "none"].itertuples():
        lines.append(f"| {row.campaign} | {row.policy} | {row.estimator} | {row.bias_ref:.5g} | "
                     f"{row.variance_emp:.5g} | {row.mse_ref:.5g} | {row.valid_repeats}/{row.repeats} |")
    for (campaign, policy), group in chosen[chosen.estimator == "DR"].groupby(["campaign", "policy"]):
        best = group.loc[group.mse_ref.idxmin()]
        dm = chosen[(chosen.campaign == campaign) & (chosen.policy == policy)
                    & (chosen.estimator == "DM") & (chosen.clip_id == "none")].iloc[0]
        ratio = best.mse_ref / dm.mse_ref if dm.mse_ref > 0 else np.nan
        lines.extend(["", f"{campaign}/{policy} 的固定网格最低 DR 参考 MSE 与同组 DM 参考 MSE 的比值为 {ratio:.4g}。"])
    lines.extend(["", "强截断削弱 DR 残差校正项，使其更多依赖奖励模型。"
                  "在参考 MSE 比值接近 1 的设置中，分解曲线与权重诊断有助于区分模型项和高波动校正项的贡献。"
                  "该解释以当前冻结策略、参考值和奖励模型为条件。"])
    lines.extend(["", "## 权重与覆盖", "",
                  "下表以各组固定网格最低 DR 参考 MSE 对应的规则为诊断对象，ESS/n、截断行比例和移除权重比例取重复均值。"
                  "最大原始权重取这些重复中观测到的最大值。", "",
                  "| Campaign | 策略 | 规则 | 原始 ESS/n | 截断 ESS/n | 截断行比例 | 移除权重比例 | 最大原始权重 |",
                  "|---|---|---|---|---|---|---|---|"])
    for (campaign, policy), group in chosen[chosen.estimator == "DR"].groupby(["campaign", "policy"]):
        best = group.loc[group.mse_ref.idxmin()]
        diagnostics = tables["diagnostics"]
        diagnostics = diagnostics[(diagnostics.campaign == campaign) & (diagnostics.policy == policy)
                                  & (diagnostics.n == maximum) & (diagnostics.clip_id == best.clip_id)]
        means = diagnostics.mean(numeric_only=True)
        lines.append(f"| {campaign} | {policy} | {best.clip_id} | {means.ess_raw / maximum:.3%} | "
                     f"{means.ess_fraction:.3%} | {means.clip_fraction:.3%} | "
                     f"{means.removed_weight_fraction:.3%} | {diagnostics.raw_max_weight.max():.5g} |")
    lines.extend(["",
                  "## 样本量、分位数与排序", "",
                  "| Campaign | 策略 | 最小 n 无截断 DR MSE | 最大 n 无截断 DR MSE | 最大 n 最低分位数规则 |",
                  "|---|---|---|---|---|"])
    observations = []
    for (campaign, policy), group in summary[(summary.estimator == "DR") & (summary.clip_id == "none")].groupby(["campaign", "policy"]):
        group = group.sort_values("n")
        quantiles = summary[(summary.campaign == campaign) & (summary.policy == policy)
                            & (summary.estimator == "DR") & (summary.n == maximum)
                            & summary.clip_id.str.startswith("qpos_")]
        quantile_text = "未启用"
        if len(quantiles):
            best = quantiles.loc[quantiles.mse_ref.idxmin()]
            quantile_text = f"{best.clip_id}, MSE={best.mse_ref:.5g}"
        lines.append(f"| {campaign} | {policy} | {group.iloc[0].mse_ref:.5g} | {group.iloc[-1].mse_ref:.5g} | {quantile_text} |")
        first, last = group.iloc[0], group.iloc[-1]
        direction = "下降" if last.mse_ref < first.mse_ref else "上升或持平"
        share = last.variance_emp / last.mse_ref if last.mse_ref > 0 else np.nan
        observations.extend(["", f"{campaign}/{policy} 的无截断 DR 参考 MSE 从 n={first.n:,} 到 n={last.n:,} "
                      f"{direction}，最大样本量下经验方差占其参考 MSE 的 {share:.1%}。"])
    lines.extend(observations)
    lines.extend(["", "各样本量分别抽样并重新拟合。罕见的大权重可能使少量重复主导平均平方误差，"
                  "应结合逐次误差和 Monte Carlo 区间解释样本量曲线。"])
    for campaign in manifest["requested"]["campaigns"]:
        rank = manifest["provenance"][campaign]["reference_rank"]
        rows = tables["ranking"]
        rows = rows[(rows.campaign == campaign) & (rows.n == maximum) & (rows.clip_id == "none")]
        text = "，".join(f"{row.estimator} {row.agreement_rate:.1%}" for row in rows.itertuples())
        lines.extend(["", f"{campaign} 的参考差为 {rank['value']:.6g}，配对 95% 区间为 "
                      f"[{rank['ci_low']:.6g}, {rank['ci_high']:.6g}]，参考排序状态为 {rank['status']}。"
                      f"无截断的经验排序一致率为 {text}。"])
    residual = (summary.mse_ref - summary.bias_ref ** 2 - summary.variance_emp).abs().max()
    diagnostics = tables["diagnostics"]
    timing = diagnostics.drop_duplicates(["campaign", "n", "repeat"])
    lines.extend(["", "## 统计口径与验收", "",
                  f"- {len(tables['replicates']):,} 条逐次估计，{len(summary):,} 条汇总，"
                  f"{len(tables['ranking']):,} 条排序汇总。",
                  f"- 偏差平方加经验方差与参考 MSE 的最大残差为 {residual:.3g}，经验方差使用分母 R。",
                  f"- 奖励模型拟合时间合计 {timing.fit_seconds.sum():.1f} 秒。"
                  f"捕获的拟合告警总数为 {timing.fit_warning_count.sum():.0f}，详情见 diagnostics.csv。",
                  f"- 无定义估计共 {tables['replicates'].estimate.isna().sum()} 条，负估计共 "
                  f"{tables['replicates'].estimate.lt(0).sum()} 条。",
                  "- 区间条件于固定日志、固定参考值和相应有效估计集合，Monte Carlo 不确定性与参考抽样误差分别保存。",
                  "- ESS 用于诊断权重集中程度。稀疏点击、低覆盖、上下文差异和记录依赖性共同限制结论范围。",
                  "- 两目标策略严格排序时 Spearman 取 +1 或 -1，报告同时提供并列率及有效配对次数。", "",
                  "## 图表资产", "",
                  "同一张图中的颜色和标记见顶部图例。阴影为固定参考值下对重复抽样计算的 95% 区间。", ""])
    campaigns = manifest["requested"]["campaigns"]
    charts = [(comparison_stem("clipping_mse", campaigns), "固定上限如何改变四种估计器的参考 MSE")]
    charts.extend((f"sample_size_{c}", f"{c.title()} 的样本量与有符号偏差、方差、参考 MSE") for c in campaigns)
    charts.extend([(comparison_stem("variance_decomposition", campaigns), "无截断时的方差柱状图，柱顶为方差与偏差平方之和"),
                   (comparison_stem("weight_diagnostics", campaigns), "有效样本量比例、截断行比例与移除的权重"),
                   (comparison_stem("ranking_agreement", campaigns), "各估计器与经验参考排序的一致率")])
    for stem, title in charts:
        relative = Path(os.path.relpath(figure_dir / stem, path.parent)).as_posix()
        lines.extend([f"**{title}**　[矢量 PDF]({relative}.pdf)", "", f"![{title}]({relative}.png)", ""])
    lines.extend(["逐阈值的偏差平方与方差、正权重分位数的 MSE 也可从同一份 CSV 单独出图。"
                  "在绘图命令后添加 `--details-dir tmp/e_figure_details` 即可。", ""])
    lines.extend([
                  "## 复现与交接", "",
                  f"逐次结果和汇总表位于 {run_dir.as_posix()}。manifest.json 记录输入、源码、环境及 CSV 的 SHA-256。"
                  f"图表为 300 DPI PNG 与矢量 PDF，见 {figure_dir.as_posix()}。"
                  "海报表格由 generated_results.tex 读取，同源数值的追溯信息在 poster/results_provenance.json。", "",
                  "使用已保存的结果重新出图，可直接在项目根目录运行以下命令。完整实验的重跑方法见 README 的成员 E 小节。", "",
                  "~~~powershell",
                  "uv sync --locked",
                  f'uv run python -m scripts.plot_sensitivity --run-dir "{run_dir.as_posix()}" --out-dir "{figure_dir.as_posix()}"',
                  "~~~", "",
                  "方法背景见 OBD/OBP 原始论文和截断、收缩的 OPE 研究。"
                  " [Saito 等](https://arxiv.org/abs/2008.07146)，"
                  "[Su 等](https://proceedings.mlr.press/v119/su20a.html)。", ""])
    # A dated audit is authored separately from the generated statistics. Carry
    # it forward only for the exact result manifest that was checked.
    audit_header = "## 本次交付验证记录\n\n<!-- run-manifest-sha256: " + file_digest(run_dir / "manifest.json") + " -->"
    if path.is_file():
        _, match, notes = path.read_text(encoding="utf-8").partition("\n" + audit_header + "\n")
        if match:
            lines.extend([audit_header, notes.rstrip(), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_run(run_dir, out_dir, poster_dir=None, report_path=None, details_dir=None):
    manifest, tables = load_run(run_dir)
    out = Path(out_dir)
    if details_dir is not None:
        details_dir = Path(details_dir)
        if details_dir.resolve() == out.resolve():
            raise ValueError("Use a separate directory for supplementary figures")
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    n = max(manifest["requested"]["sample_sizes"])
    r = manifest["requested"]["repeats"]
    campaigns = manifest["requested"]["campaigns"]
    sources = ", ".join(sorted({manifest["provenance"][c]["source"] for c in campaigns}))
    caption = f"{sources.title()} logs | R={r}, n={n:,} | Fixed Random IPS reference."
    plot_clipping(tables["summary"], campaigns, n, out, caption, files)
    for campaign in campaigns:
        source = manifest["provenance"][campaign]["source"]
        plot_sizes(tables["summary"], campaign, out, f"{campaign.title()} | {source} logs | R={r} | Fixed Random IPS reference.", files)
    plot_variance(tables["summary"], campaigns, n, out, caption, files)
    plot_weight_diagnostics(tables["diagnostics"], campaigns, n, out, caption, files)
    plot_rankings(tables["ranking"], campaigns, n, out, caption, files)
    if details_dir is not None:
        details_dir.mkdir(parents=True, exist_ok=True)
        details = {}
        for campaign in campaigns:
            plot_clipping_details(tables["summary"], campaign, n, details_dir, f"{campaign.title()} | {caption}", details)
        write_figure_manifest(details_dir, manifest, details)
    if poster_dir:
        poster_assets(manifest, tables, Path(poster_dir))
    if report_path:
        write_report(manifest, tables, Path(report_path), Path(run_dir), out)
    write_figure_manifest(out, manifest, files)


def write_figure_manifest(out, manifest, files):
    (out / "manifest.json").write_text(json.dumps(dict(
        run_id=manifest["run_id"], font="DejaVu Sans", png_dpi=300,
        input_csv_sha256=manifest["output_sha256"], output_sha256=files,
        plotting_source_sha256=file_digest(Path(__file__)),
    ), indent=2) + "\n", encoding="utf-8")
