# Changelog

## 0.4.0 — defensibility and held-out evaluation pass

- Replaced the ambiguous 15-row sample with a fully synthetic eight-team generator and checked-in offline sample.
- Replaced iterative strength scaling with a sum-to-zero maximum-likelihood attack/defence Poisson model.
- Added chronological train/test evaluation; Dixon–Coles `rho` is fitted on training matches only.
- Removed score clipping from observed-score likelihood calculations.
- Added outcome log loss, multiclass Brier score, accuracy and a training-only empirical baseline.
- Added de-margined bookmaker scoring, preferring closing Bet365 columns when available.
- Added a 120-world known-truth robustness study over two sample sizes and two true low-score-dependence regimes.
- Retired the old in-sample `-4.209 vs -4.221` Dixon–Coles headline.
- Added score/result validation and shared ranks for exact points/GD/GF ties instead of silently inventing an alphabetical tiebreak.
- Hardened the CSV downloader and FBref parser; the latter selects by expected table schema rather than table size.
- Added `docs/METHODOLOGY.md`, two diagnostic figures and expanded tests from 3 to 12.

## 0.3.0 — update round 7

Added the Dixon–Coles low-score correction: tau reweights the four lowest scorelines, rho is fitted by profile likelihood, and the summary reports scoreline log likelihood with and without the correction alongside the outcome comparison against the market.

## 0.2.0 — update round 3

Added a Poisson goals model with attack and defence strengths and home advantage, producing match outcome probabilities and a log likelihood comparison against bookmaker implied probabilities where odds columns exist.

## 0.1.0

Initial release.
