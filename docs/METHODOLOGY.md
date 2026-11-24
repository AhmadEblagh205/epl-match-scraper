# Methodology and audit notes

## 1. Scope

This repository has three deliberately separated jobs:

1. acquire completed match results;
2. transform them into validated standings-ready data;
3. evaluate a compact football score model.

The acquisition layer is not evidence for the statistical model, and the statistical model is not presented as an automated betting strategy.

## 2. Input contract

Core columns are:

`Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR`.

Validation fails if dates cannot be parsed, club names are blank, teams play themselves, goals are negative/non-integer, `FTR` is outside `H/D/A`, or the stated result disagrees with the score.

The sample CSV is generated locally and is entirely synthetic.

## 3. League-table semantics

For ordinary standings display, teams are ordered by points, goal difference and goals scored. Exact ties receive a shared rank. Alphabetical order is used only to make the rendered DataFrame deterministic inside a shared rank.

This is intentional. A partial-season standings script should not invent a hidden fourth tiebreak. Final Premier League competition rules contain additional head-to-head and play-off provisions that are outside this small engine.

## 4. Base Poisson model

For home team `i` and away team `j`:

`log(lambda_home) = intercept + home_advantage + attack[i] + defence[j]`

`log(lambda_away) = intercept + attack[j] + defence[i]`

`defence` is a conceding/weakness effect. Positive means more expected goals conceded, not “better defence”.

Attack and defence effects each satisfy a sum-to-zero constraint. The final team effect is reconstructed as minus the sum of the free effects, removing the additive identifiability ambiguity.

Parameters are estimated by maximum likelihood with L-BFGS-B. The goal-count likelihood uses the exact Poisson log PMF through `gammaln` rather than a truncated score grid.

## 5. Dixon–Coles correction

The standard low-score factor is applied to only four cells:

- 0–0: `1 - lambda_home * lambda_away * rho`
- 0–1: `1 + lambda_home * rho`
- 1–0: `1 + lambda_away * rho`
- 1–1: `1 - rho`

All other scorelines have factor 1.

The independent Poisson strengths are fitted first. `rho` is then profiled on the training set within a positivity-preserving interval capped at ±0.20. The report flags estimates near that cap because a boundary estimate is a warning about weak information, not a badge of model strength.

## 6. Outcome probabilities

A 0–12 by 0–12 score grid is sufficient for football-scale means; the tiny remaining tail is renormalised only for the 1X2 probability calculation. Observed scoreline log loss itself uses the analytical Poisson mass and therefore does not clip an observed 9+ goal score into an 8-goal bin, as the old implementation did.

## 7. Chronological evaluation

The repository no longer fits and scores the same rows.

Rows are sorted by date. A requested training fraction defines the initial split, moved forward only if necessary so every evaluation club has appeared at least once in training. Team parameters and Dixon–Coles `rho` are estimated from training rows only.

Held-out metrics are:

- scoreline log loss;
- 1X2 log loss;
- multiclass Brier score;
- modal 1X2 accuracy.

The no-feature comparator is the H/D/A frequency distribution estimated on training rows only.

## 8. Bookmaker benchmark

If suitable odds exist, inverse odds are normalised so the three implied probabilities sum to one before scoring. This removes the quoted 1X2 overround mechanically. Closing Bet365 columns are preferred when present.

This is a benchmark, not an efficiency theorem. Different books/times can have different information sets and margins.

## 9. Known-truth experiment

The synthetic generator uses eight invented teams with fixed attack/conceding effects and a known home advantage. It can sample either independent-Poisson scores or a Dixon–Coles low-score dependence.

The robustness study varies two quantities:

- information: 56 vs 224 training matches;
- truth: `rho=0` vs deliberately planted `rho=-0.12`.

Thirty independent worlds per condition are used in the checked-in report.

The central outcome is `independent test NLL - Dixon-Coles test NLL`; positive values mean the extra parameter helped on unseen matches.

This experiment is designed to test **model complexity under known truth**. It is not calibrated to the empirical Premier League distribution.

## 10. Audit decisions

The v0.4 pass changed several claims:

- retired the tiny-sample in-sample `scoreline LL -4.209 vs -4.221` headline;
- replaced the iterative strength heuristic with an identifiable MLE;
- removed score clipping from likelihood evaluation;
- moved all headline model comparison to held-out matches;
- replaced the ambiguous old sample with genuinely synthetic clubs/results;
- stopped calling points/GD/GF sorting “complete tiebreaking”;
- made the HTML parser search for the expected fixture schema rather than the largest page table.

## 11. What would strengthen this next

A serious empirical extension would use multiple historical seasons with a predeclared rolling protocol, promoted-team priors and time decay, then compare independent Poisson, Dixon–Coles and richer xG/event models on future seasons. Any bookmaker comparison should pin the exact odds timestamp/source and remain strictly out of sample.
