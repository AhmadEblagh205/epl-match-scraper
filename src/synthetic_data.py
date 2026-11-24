"""Deterministic synthetic football data for offline tests and demonstrations.

These teams, scores and optional bookmaker-style probabilities are simulated.
They are not Premier League results and are not evidence about real teams.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import exp, factorial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEAMS = [
    "Ashford", "Briar FC", "Cedar United", "Dunwich", "Elm City",
    "Foxbridge", "Glen Athletic", "Harbour Town",
]


@dataclass(frozen=True)
class SyntheticTruth:
    intercept: float
    home_advantage: float
    attack: dict[str, float]
    defence: dict[str, float]
    rho: float


def _tau(i: int, j: int, mu_h: float, mu_a: float, rho: float) -> float:
    if i == 0 and j == 0:
        return 1 - mu_h * mu_a * rho
    if i == 0 and j == 1:
        return 1 + mu_h * rho
    if i == 1 and j == 0:
        return 1 + mu_a * rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0


def _sample_score(rng: np.random.Generator, mu_h: float, mu_a: float,
                  rho: float, max_goals: int = 12) -> tuple[int, int]:
    ph = np.array([exp(-mu_h) * mu_h ** k / factorial(k)
                   for k in range(max_goals + 1)])
    pa = np.array([exp(-mu_a) * mu_a ** k / factorial(k)
                   for k in range(max_goals + 1)])
    grid = np.outer(ph, pa)
    for i in range(2):
        for j in range(2):
            grid[i, j] *= _tau(i, j, mu_h, mu_a, rho)
    if (grid < 0).any():
        raise ValueError("rho makes a Dixon-Coles low-score probability negative")
    grid /= grid.sum()
    flat = rng.choice(grid.size, p=grid.ravel())
    return tuple(map(int, np.unravel_index(flat, grid.shape)))


def generate_matches(*, seed: int = 2026, cycles: int = 4,
                     rho: float = -0.07,
                     teams: list[str] | None = None) -> tuple[pd.DataFrame, SyntheticTruth]:
    teams = list(teams or DEFAULT_TEAMS)
    n = len(teams)
    if n < 4:
        raise ValueError("at least four teams are required")
    rng = np.random.default_rng(seed)

    raw_attack = np.linspace(0.34, -0.34, n)
    raw_defence = np.array([0.20, 0.10, -0.05, -0.20, 0.16, -0.11, 0.02, -0.12])
    if n != len(raw_defence):
        raw_defence = np.linspace(0.20, -0.20, n)
    attack = raw_attack - raw_attack.mean()
    defence = raw_defence - raw_defence.mean()  # positive = more goals conceded
    intercept = float(np.log(1.25))
    home_adv = float(np.log(1.24))
    truth = SyntheticTruth(
        intercept=intercept,
        home_advantage=home_adv,
        attack=dict(zip(teams, attack)),
        defence=dict(zip(teams, defence)),
        rho=float(rho),
    )

    base = [(h, a) for h in teams for a in teams if h != a]
    rows = []
    date = pd.Timestamp("2024-08-01")
    for cycle in range(cycles):
        fixtures = base.copy()
        rng.shuffle(fixtures)
        for h, a in fixtures:
            mu_h = np.exp(intercept + home_adv + truth.attack[h] + truth.defence[a])
            mu_a = np.exp(intercept + truth.attack[a] + truth.defence[h])
            gh, ga = _sample_score(rng, float(mu_h), float(mu_a), rho)
            rows.append({
                "Date": date.strftime("%Y-%m-%d"),
                "HomeTeam": h,
                "AwayTeam": a,
                "FTHG": gh,
                "FTAG": ga,
                "FTR": "H" if gh > ga else "A" if ga > gh else "D",
            })
            date += pd.Timedelta(days=1)
    return pd.DataFrame(rows), truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--rho", type=float, default=-0.07)
    ap.add_argument("--output", default=str(ROOT / "data" / "sample" / "sample_matches.csv"))
    args = ap.parse_args()
    df, truth = generate_matches(seed=args.seed, cycles=args.cycles, rho=args.rho)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} synthetic matches -> {out}")
    print(f"Known truth: home log-advantage={truth.home_advantage:.3f}, rho={truth.rho:.3f}")


if __name__ == "__main__":
    main()
