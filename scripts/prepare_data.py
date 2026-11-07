"""下载 + 清洗 + 切分 + 特征工程，一步完成。

    uv run python scripts/prepare_data.py                   # 全量数据（约 413 MB zip，首次下载较慢）
    uv run python scripts/prepare_data.py --sample          # 官方 1 万行样例，几秒跑完（MVP / 调试）
    uv run python scripts/prepare_data.py --campaigns men --max-rows 200000
    uv run python scripts/prepare_data.py --skip-download   # 数据已在本地，只重新构建

输出 data/processed/<campaign>/（格式见 ope/data/dataset.py）和 data_summary.csv。
"""

import argparse
import logging

import pandas as pd

from ope.data import DataConfig, build_campaign, download_full, download_sample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sample", action="store_true", help="使用官方 1 万行样例")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--campaigns", nargs="+", default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--processed-dir", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = DataConfig.from_yaml(args.config, campaigns=args.campaigns, max_rows=args.max_rows)
    if args.sample:
        cfg.use_sample = True
        cfg.processed_dir = args.processed_dir or cfg.processed_dir + "_sample"
    elif args.processed_dir:
        cfg.processed_dir = args.processed_dir

    if not args.skip_download:
        download_sample(cfg.sample_dir) if args.sample else download_full(cfg.raw_dir)

    rows = []
    for campaign in cfg.campaigns:
        meta = build_campaign(cfg, campaign)
        for policy, splits in meta["splits"].items():
            rep = meta["cleaning"][policy]
            for split, s in splits.items():
                rows.append({
                    "campaign": campaign, "policy": policy, "split": split,
                    "n_rounds": s["n_rounds"], "n_clicks": s["n_clicks"], "ctr": s["ctr"],
                    "time_start": s["time_start"], "time_end": s["time_end"],
                    "n_actions": meta["n_actions"], "context_dim": meta["context_dim"],
                    "pscore_min": rep["pscore"]["min"], "pscore_max": rep["pscore"]["max"],
                    "rows_dropped_in_cleaning": rep["n_raw"] - rep["n_clean"],
                })
    summary = pd.DataFrame(rows)
    out = f"{cfg.processed_dir}/data_summary.csv"
    summary.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.drop(columns=["time_start", "time_end"]).to_string(index=False))
    print(f"\n汇总 -> {out}")


if __name__ == "__main__":
    main()
