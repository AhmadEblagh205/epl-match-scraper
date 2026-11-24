"""Known-truth study: when does the Dixon-Coles extra parameter generalise?"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from poisson_model import (chronological_split, fit_dixon_coles_rho,
                           fit_independent_poisson, scoreline_log_loss)
from synthetic_data import generate_matches

ROOT = Path(__file__).resolve().parents[1]


def run_study(n_worlds: int = 30,
              cycles_values: tuple[int, ...] = (2, 8),
              truth_rhos: tuple[float, ...] = (0.0, -0.12)) -> tuple[pd.DataFrame, dict]:
    """Repeat fit/test comparisons under known low-score dependence.

    `cycles=2` gives 112 total matches (56 fit / 56 test), while `cycles=8`
    gives 448 total matches (224 / 224) for the default eight-team world.
    The rho=-0.12 scenario is deliberately stronger than the sample fixture so
    the role of data volume and true low-score dependence can be seen clearly.
    """
    rows = []
    for cycles in cycles_values:
        for rho_true in truth_rhos:
            for seed in range(n_worlds):
                matches, _ = generate_matches(seed=10_000 + seed,
                                              cycles=cycles,
                                              rho=rho_true)
                train, test = chronological_split(matches, 0.5)
                fit = fit_independent_poisson(train)
                fit_dixon_coles_rho(train, fit)
                tr_ind = scoreline_log_loss(train, fit, 0.0)
                tr_dc = scoreline_log_loss(train, fit, fit.rho)
                te_ind = scoreline_log_loss(test, fit, 0.0)
                te_dc = scoreline_log_loss(test, fit, fit.rho)
                rows.append({
                    "cycles": cycles,
                    "n_train": len(train),
                    "n_test": len(test),
                    "truth_rho": rho_true,
                    "seed": seed,
                    "fitted_rho": fit.rho,
                    "rho_at_boundary": fit.rho_at_boundary,
                    "train_gain_ind_minus_dc": tr_ind - tr_dc,
                    "test_gain_ind_minus_dc": te_ind - te_dc,
                })
    df = pd.DataFrame(rows)
    summary = {}
    for (cycles, rho_true), g in df.groupby(["cycles", "truth_rho"], sort=True):
        key = f"train{int(g.n_train.iloc[0])}_rho{rho_true:+.2f}"
        summary[key] = {
            "n_worlds": int(len(g)),
            "n_train": int(g.n_train.iloc[0]),
            "n_test": int(g.n_test.iloc[0]),
            "truth_rho": float(rho_true),
            "median_fitted_rho": round(float(g.fitted_rho.median()), 4),
            "median_train_nll_gain": round(float(g.train_gain_ind_minus_dc.median()), 5),
            "median_test_nll_gain": round(float(g.test_gain_ind_minus_dc.median()), 5),
            "dc_better_test_fraction": round(float((g.test_gain_ind_minus_dc > 0).mean()), 3),
            "boundary_fraction": round(float(g.rho_at_boundary.mean()), 3),
        }
    return df, summary


def save_figures(df: pd.DataFrame) -> None:
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    configs = list(df.groupby(["n_train", "truth_rho"], sort=True))
    labels = [f"n={n}\nrho={rho:.2f}" for (n, rho), _ in configs]
    train_data = [g.train_gain_ind_minus_dc.to_numpy() for _, g in configs]
    test_data = [g.test_gain_ind_minus_dc.to_numpy() for _, g in configs]
    rho_data = [g.fitted_rho.to_numpy() for _, g in configs]

    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.boxplot(train_data, positions=x - 0.16, widths=0.24,
               showfliers=False)
    ax.boxplot(test_data, positions=x + 0.16, widths=0.24,
               showfliers=False)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Independent NLL - Dixon-Coles NLL\n(positive = DC better)")
    ax.set_title("Extra low-score flexibility: training gain vs held-out value")
    ax.text(0.01, 0.98, "left box = training   right box = held-out",
            transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(fig_dir / "dc_generalisation.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.boxplot(rho_data, positions=x, widths=0.45, showfliers=False)
    for i, ((_, rho_true), _) in enumerate(configs):
        ax.plot([i - 0.27, i + 0.27], [rho_true, rho_true], linewidth=2)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Fitted Dixon-Coles rho")
    ax.set_title("Low-score dependence recovery improves with information")
    fig.tight_layout()
    fig.savefig(fig_dir / "rho_recovery.png", dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=30)
    args = ap.parse_args()
    df, summary = run_study(n_worlds=args.worlds)
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "dc_robustness_worlds.csv", index=False)
    (out_dir / "dc_robustness_summary.json").write_text(json.dumps(summary, indent=2))
    save_figures(df)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
