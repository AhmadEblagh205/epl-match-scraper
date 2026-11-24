"""Poisson / Dixon-Coles football score model with chronological evaluation.

Model:
    log(lambda_home) = intercept + home_adv + attack_home + defence_away
    log(lambda_away) = intercept + attack_away + defence_home

The defence term is a *conceding* effect: larger values mean more goals allowed.
Attack and defence effects are each constrained to sum to zero for
identifiability.  Dixon-Coles rho is fitted on the training set only.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import exp, factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln

from data_validation import ensure_probability_rows, validate_matches

ROOT = Path(__file__).resolve().parents[1]
OUTCOME_COLUMNS = ["p_home", "p_draw", "p_away"]
OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


@dataclass
class PoissonFit:
    teams: list[str]
    intercept: float
    home_advantage: float
    attack: dict[str, float]
    defence: dict[str, float]
    rho: float = 0.0
    converged: bool = True
    nll: float | None = None
    rho_at_boundary: bool = False


def _effects(free: np.ndarray) -> np.ndarray:
    return np.r_[free, -float(np.sum(free))]


def _unpack(theta: np.ndarray, teams: list[str]):
    n = len(teams)
    intercept = float(theta[0])
    home_adv = float(theta[1])
    attack = _effects(theta[2:2 + n - 1])
    defence = _effects(theta[2 + n - 1:2 + 2 * (n - 1)])
    return intercept, home_adv, attack, defence


def fit_independent_poisson(matches: pd.DataFrame) -> PoissonFit:
    m = validate_matches(matches).sort_values("Date").reset_index(drop=True)
    teams = sorted(set(m.HomeTeam) | set(m.AwayTeam))
    if len(teams) < 2:
        raise ValueError("at least two teams are required")
    idx = {t: i for i, t in enumerate(teams)}
    hi = m.HomeTeam.map(idx).to_numpy()
    ai = m.AwayTeam.map(idx).to_numpy()
    gh = m.FTHG.to_numpy(dtype=float)
    ga = m.FTAG.to_numpy(dtype=float)

    mean_away = max(float(np.mean(ga)), 0.2)
    mean_home = max(float(np.mean(gh)), 0.2)
    x0 = np.zeros(2 + 2 * (len(teams) - 1), dtype=float)
    x0[0] = np.log((mean_home + mean_away) / 2)
    x0[1] = np.log(mean_home / mean_away)

    def nll(theta: np.ndarray) -> float:
        intercept, home_adv, attack, defence = _unpack(theta, teams)
        log_h = intercept + home_adv + attack[hi] + defence[ai]
        log_a = intercept + attack[ai] + defence[hi]
        # Guard the optimiser from absurd exponentiation while preserving a
        # broad region for plausible football goal means.
        if np.max(np.abs(np.r_[log_h, log_a])) > 7:
            return 1e12
        mu_h = np.exp(log_h)
        mu_a = np.exp(log_a)
        val = np.sum(mu_h - gh * log_h + gammaln(gh + 1))
        val += np.sum(mu_a - ga * log_a + gammaln(ga + 1))
        return float(val)

    res = minimize(nll, x0, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-11})
    if not res.success:
        raise RuntimeError(f"Poisson optimisation failed: {res.message}")
    intercept, home_adv, a, d = _unpack(res.x, teams)
    return PoissonFit(
        teams=teams,
        intercept=intercept,
        home_advantage=home_adv,
        attack=dict(zip(teams, map(float, a))),
        defence=dict(zip(teams, map(float, d))),
        nll=float(res.fun),
    )


def expected_goals(fit: PoissonFit, home: str, away: str) -> tuple[float, float]:
    if home not in fit.attack or away not in fit.attack:
        missing = [t for t in (home, away) if t not in fit.attack]
        raise ValueError(f"unseen team(s) in prediction: {missing}")
    mu_h = np.exp(fit.intercept + fit.home_advantage
                  + fit.attack[home] + fit.defence[away])
    mu_a = np.exp(fit.intercept + fit.attack[away] + fit.defence[home])
    return float(mu_h), float(mu_a)


def dc_tau(i: int, j: int, mu_h: float, mu_a: float, rho: float) -> float:
    if i == 0 and j == 0:
        return 1 - mu_h * mu_a * rho
    if i == 0 and j == 1:
        return 1 + mu_h * rho
    if i == 1 and j == 0:
        return 1 + mu_a * rho
    if i == 1 and j == 1:
        return 1 - rho
    return 1.0


def _rho_bounds(matches: pd.DataFrame, fit: PoissonFit,
                hard_bound: float = 0.20) -> tuple[float, float]:
    mus = [expected_goals(fit, r.HomeTeam, r.AwayTeam)
           for r in matches.itertuples(index=False)]
    lower = max(max(-1 / mh, -1 / ma) for mh, ma in mus) + 1e-6
    upper = min(min(1 / (mh * ma) for mh, ma in mus), 1.0) - 1e-6
    return max(lower, -hard_bound), min(upper, hard_bound)


def fit_dixon_coles_rho(matches: pd.DataFrame, fit: PoissonFit,
                        hard_bound: float = 0.20) -> PoissonFit:
    m = validate_matches(matches)
    lo, hi = _rho_bounds(m, fit, hard_bound=hard_bound)
    if not lo < hi:
        raise ValueError("no valid Dixon-Coles rho interval for these means")

    def objective(rho: float) -> float:
        total = 0.0
        for r in m.itertuples(index=False):
            if r.FTHG <= 1 and r.FTAG <= 1:
                mh, ma = expected_goals(fit, r.HomeTeam, r.AwayTeam)
                tau = dc_tau(int(r.FTHG), int(r.FTAG), mh, ma, float(rho))
                if tau <= 0:
                    return 1e12
                total -= np.log(tau)
        return float(total)

    res = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-7})
    rho = float(res.x)
    boundary_tol = max(2e-3, 0.02 * (hi - lo))
    fit.rho = rho
    fit.rho_at_boundary = min(abs(rho - lo), abs(rho - hi)) <= boundary_tol
    return fit


def score_grid(mu_h: float, mu_a: float, rho: float = 0.0,
               max_goals: int = 12) -> np.ndarray:
    if mu_h <= 0 or mu_a <= 0:
        raise ValueError("expected goals must be positive")
    ph = np.array([exp(-mu_h) * mu_h ** k / factorial(k)
                   for k in range(max_goals + 1)])
    pa = np.array([exp(-mu_a) * mu_a ** k / factorial(k)
                   for k in range(max_goals + 1)])
    grid = np.outer(ph, pa)
    for i in range(2):
        for j in range(2):
            grid[i, j] *= dc_tau(i, j, mu_h, mu_a, rho)
    if (grid < -1e-12).any():
        raise ValueError("rho creates a negative low-score probability")
    grid = np.maximum(grid, 0)
    # Tail beyond max_goals is tiny at football-scale means.  Renormalisation
    # keeps outcome probabilities exactly coherent while the scoreline LL below
    # uses the untruncated analytical Poisson mass for observed scores.
    total = grid.sum()
    if total <= 0:
        raise ValueError("invalid score probability grid")
    return grid / total


def match_probs(mu_h: float, mu_a: float, rho: float = 0.0,
                max_goals: int = 12) -> tuple[float, float, float]:
    grid = score_grid(mu_h, mu_a, rho=rho, max_goals=max_goals)
    home = float(np.tril(grid, k=-1).sum())  # row=home goals, col=away goals
    draw = float(np.trace(grid))
    away = 1.0 - home - draw
    return home, draw, float(away)


def predict(matches: pd.DataFrame, fit: PoissonFit,
            *, rho: float | None = None) -> pd.DataFrame:
    m = validate_matches(matches)
    rho = fit.rho if rho is None else float(rho)
    rows = []
    for r in m.itertuples(index=False):
        mu_h, mu_a = expected_goals(fit, r.HomeTeam, r.AwayTeam)
        h, d, a = match_probs(mu_h, mu_a, rho=rho)
        rows.append({"Date": r.Date, "HomeTeam": r.HomeTeam,
                     "AwayTeam": r.AwayTeam, "FTR": r.FTR,
                     "mu_home": mu_h, "mu_away": mu_a,
                     "p_home": h, "p_draw": d, "p_away": a})
    out = pd.DataFrame(rows)
    ensure_probability_rows(out[OUTCOME_COLUMNS].to_numpy())
    return out


def outcome_metrics(preds: pd.DataFrame, matches: pd.DataFrame) -> dict[str, float]:
    p = preds[OUTCOME_COLUMNS].to_numpy(float)
    y = np.array([OUTCOME_INDEX[x] for x in matches.FTR], dtype=int)
    chosen = p[np.arange(len(p)), y]
    onehot = np.eye(3)[y]
    return {
        "log_loss": float(-np.mean(np.log(np.clip(chosen, 1e-15, 1)))),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "accuracy": float(np.mean(np.argmax(p, axis=1) == y)),
    }


def baseline_metrics(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    counts = train.FTR.value_counts(normalize=True)
    probs = np.array([counts.get("H", 0.0), counts.get("D", 0.0), counts.get("A", 0.0)])
    probs = np.clip(probs, 1e-8, None); probs /= probs.sum()
    preds = pd.DataFrame(np.tile(probs, (len(test), 1)), columns=OUTCOME_COLUMNS)
    return outcome_metrics(preds, test)


def market_metrics(matches: pd.DataFrame) -> tuple[dict[str, float], str] | None:
    choices = [
        (("B365CH", "B365CD", "B365CA"), "Bet365 closing 1X2"),
        (("B365H", "B365D", "B365A"), "Bet365 1X2 snapshot"),
    ]
    for cols, label in choices:
        if set(cols).issubset(matches.columns):
            odds = matches[list(cols)].apply(pd.to_numeric, errors="coerce")
            valid = odds.notna().all(axis=1) & (odds > 1).all(axis=1)
            if not valid.all():
                return None
            inv = 1.0 / odds.to_numpy(float)
            p = inv / inv.sum(axis=1, keepdims=True)
            preds = pd.DataFrame(p, columns=OUTCOME_COLUMNS)
            return outcome_metrics(preds, matches), label
    return None


def scoreline_log_loss(matches: pd.DataFrame, fit: PoissonFit,
                       rho: float | None = None) -> float:
    m = validate_matches(matches)
    rho = fit.rho if rho is None else float(rho)
    ll = 0.0
    for r in m.itertuples(index=False):
        mu_h, mu_a = expected_goals(fit, r.HomeTeam, r.AwayTeam)
        ll += -mu_h + r.FTHG * np.log(mu_h) - gammaln(r.FTHG + 1)
        ll += -mu_a + r.FTAG * np.log(mu_a) - gammaln(r.FTAG + 1)
        tau = dc_tau(int(r.FTHG), int(r.FTAG), mu_h, mu_a, rho)
        if tau <= 0:
            return float("inf")
        ll += np.log(tau)
    return float(-ll / len(m))


def chronological_split(matches: pd.DataFrame, train_fraction: float = 0.5):
    if not 0.2 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must lie in [0.2, 0.9]")
    m = validate_matches(matches).sort_values("Date").reset_index(drop=True)
    split = max(1, int(round(len(m) * train_fraction)))
    all_teams = set(m.HomeTeam) | set(m.AwayTeam)
    # Move the split forward only if needed to ensure every evaluation team has
    # at least appeared in the fit sample.
    while split < len(m):
        tr = m.iloc[:split]
        seen = set(tr.HomeTeam) | set(tr.AwayTeam)
        if all_teams <= seen:
            break
        split += 1
    if split >= len(m):
        raise ValueError("cannot create a held-out split after all teams are observed")
    return m.iloc[:split].copy(), m.iloc[split:].copy()


def evaluate_chronological(matches: pd.DataFrame, train_fraction: float = 0.5) -> dict:
    train, test = chronological_split(matches, train_fraction=train_fraction)
    fit = fit_independent_poisson(train)
    fit_dixon_coles_rho(train, fit)

    pred_ind = predict(test, fit, rho=0.0)
    pred_dc = predict(test, fit, rho=fit.rho)
    result = {
        "n_matches": int(len(train) + len(test)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "split_date": str(test.Date.iloc[0].date()),
        "home_log_advantage": round(fit.home_advantage, 4),
        "home_goal_multiplier": round(float(np.exp(fit.home_advantage)), 4),
        "dixon_coles_rho": round(fit.rho, 4),
        "rho_at_search_boundary": bool(fit.rho_at_boundary),
        "train_scoreline_log_loss_independent": round(scoreline_log_loss(train, fit, 0.0), 4),
        "train_scoreline_log_loss_dc": round(scoreline_log_loss(train, fit, fit.rho), 4),
        "test_scoreline_log_loss_independent": round(scoreline_log_loss(test, fit, 0.0), 4),
        "test_scoreline_log_loss_dc": round(scoreline_log_loss(test, fit, fit.rho), 4),
        "test_outcome_independent": {k: round(v, 4) for k, v in outcome_metrics(pred_ind, test).items()},
        "test_outcome_dc": {k: round(v, 4) for k, v in outcome_metrics(pred_dc, test).items()},
        "test_outcome_empirical_baseline": {k: round(v, 4) for k, v in baseline_metrics(train, test).items()},
        "top_attack": max(fit.attack, key=fit.attack.get),
        "lowest_conceding_effect": min(fit.defence, key=fit.defence.get),
    }
    market = market_metrics(test)
    if market:
        metrics, label = market
        result["market_benchmark"] = {"source": label, **{k: round(v, 4) for k, v in metrics.items()}}
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--train-fraction", type=float, default=0.5)
    args = ap.parse_args()
    matches = pd.read_csv(args.csv)
    result = evaluate_chronological(matches, train_fraction=args.train_fraction)
    out = ROOT / "reports" / "poisson_summary.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
