"""Render validated experiment CSVs as PDF/PNG and optional report/Poster assets."""
import argparse
from pathlib import Path

from ope.experiments.plotting import plot_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("results/sensitivity/e_full"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures/e_full"))
    parser.add_argument("--poster-assets-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--details-dir", type=Path,
                        help="Also export per-campaign decomposition and quantile curves to this directory")
    args = parser.parse_args()
    plot_run(args.run_dir, args.out_dir, args.poster_assets_dir, args.report, args.details_dir)
    print(f"Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
