"""计算 OPE 基准的 Ground Truth。

    # 1) 两个线上策略（Random、Bernoulli TS）的真实值
    uv run python scripts/ground_truth.py

    # 2) 再加上成员 C 新策略的真实值（在 random/test 日志上做无偏 IPS）
    uv run python scripts/ground_truth.py --campaign all \
        --action-dist linucb=outputs/linucb_all.npy logisticts=outputs/logisticts_all.npy

每个 .npy 是在 load_bandit_feedback(campaign, "random", "test")["context"] 上逐行计算的
π_e(a|x, pos)，形状 (n_rounds, n_actions, 3) 或 (n_rounds, n_actions)。

输出 results/ground_truth/{ground_truth.csv, ground_truth.json, ctr_by_position.csv}。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ope.data import load_bandit_feedback, load_meta, on_policy_value, policy_value_from_random_logs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sample", action="store_true", help="读取 data/processed_sample")
    ap.add_argument("--campaigns", nargs="+", default=None, help="默认：所有已构建的 campaign")
    ap.add_argument("--campaign", default=None, help="--action-dist 对应的 campaign")
    ap.add_argument("--split", default="test")
    ap.add_argument("--action-dist", nargs="*", default=[], metavar="NAME=PATH")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gt = cfg.get("ground_truth", {})
    alpha, n_boot = gt.get("alpha", 0.05), gt.get("n_bootstrap", 1000)
    seed = cfg["data"].get("seed", 12345)
    suffix = "_sample" if args.sample else ""
    pdir = Path(cfg["data"]["processed_dir"] + suffix)
    out = Path(args.out or gt.get("out_dir", "results/ground_truth") + suffix)
    campaigns = args.campaigns or sorted(p.name for p in pdir.glob("*") if (p / "meta.json").exists())
    if not campaigns:
        sys.exit(f"{pdir} 下没有处理好的数据，请先运行 scripts/prepare_data.py")
    out.mkdir(parents=True, exist_ok=True)

    rows, pos_rows = [], []
    for c in campaigns:
        for policy in load_meta(c, pdir)["policies"]:
            bf = load_bandit_feedback(c, policy, args.split, pdir, mmap=True)
            if policy == "random" and not np.allclose(bf["pscore"], 1.0 / bf["n_actions"]):
                raise RuntimeError(f"{c}/random 日志的倾向得分不是均匀的，请检查原始数据")
            rows.append({"campaign": c, "policy": policy, "type": "on_policy", "split": args.split,
                         **on_policy_value(bf["reward"], alpha, n_boot, seed)})
            for k in range(bf["len_list"]):
                m = bf["position"] == k
                pos_rows.append({"campaign": c, "policy": policy, "position": k,
                                 "n_rounds": int(m.sum()), "ctr": float(bf["reward"][m].mean())})

    if args.action_dist:  # 成员 C 的新策略
        if not args.campaign:
            sys.exit("使用 --action-dist 时必须指定 --campaign")
        bf = load_bandit_feedback(args.campaign, "random", args.split, pdir)
        for spec in args.action_dist:
            name, path = spec.split("=", 1)
            rows.append({"campaign": args.campaign, "policy": name, "type": "ips_on_random_logs",
                         "split": args.split,
                         **policy_value_from_random_logs(bf, np.load(path), alpha, n_boot, seed)})

    df = pd.DataFrame(rows)
    base = df[(df.policy == "random") & (df.type == "on_policy")].set_index("campaign")["value"]
    df["lift_vs_random"] = df["value"] / df["campaign"].map(base) - 1
    df.to_csv(out / "ground_truth.csv", index=False)
    pd.DataFrame(pos_rows).to_csv(out / "ctr_by_position.csv", index=False)
    nested: dict = {}
    for r in df.to_dict("records"):
        nested.setdefault(r.pop("campaign"), {})[r.pop("policy")] = r
    with open(out / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(nested, f, ensure_ascii=False, indent=1)

    show = ["campaign", "policy", "type", "n_rounds", "value", "ci_low", "ci_high", "lift_vs_random"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.6f}".format):
        print(df[show].to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
