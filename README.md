# Premier League Data & Poisson Evaluation Laboratory

[![CI](https://github.com/AhmadEblagh205/epl-match-scraper/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/epl-match-scraper/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A small football-data engineering and statistical-modelling project: collect completed Premier League results through a stable CSV route or an optional HTML scraper, validate them into one schema, build standings, and evaluate an attack/defence Poisson model without scoring it on the same matches used to fit it.

The central v0.4 lesson is not that a more complicated football model always wins. It is that **extra flexibility should earn itself on held-out matches**. A Dixon–Coles low-score parameter always has an opportunity to improve training likelihood; in controlled repeated worlds it only delivers a reliable held-out benefit when low-score dependence is genuinely present *and* there is enough information to estimate it.

## Data routes

### Preferred: published CSV

`download_football_data.py` downloads Premier League season files from [football-data.co.uk](https://www.football-data.co.uk/). The downloader validates the season code, HTTP response and core result columns before saving anything.

```bash
python src/download_football_data.py --seasons 2324 2425
```

When closing Bet365 1X2 odds (`B365CH/B365CD/B365CA`) are present, the evaluation code uses them in preference to the non-closing `B365H/B365D/B365A` columns. In either case inverse odds are normalised row by row to remove the bookmaker overround before log loss or Brier score is calculated.

### Optional: FBref HTML

`scrape_fbref.py` exists to show the less stable route. It identifies a table by the required `Date/Home/Away/Score` columns rather than simply taking the largest HTML table. It sends a descriptive user agent and supports a delay after the request. **Check FBref's current robots.txt and terms before running it.** If a published CSV already contains the data you need, prefer the CSV.

```bash
python src/scrape_fbref.py --season 2024-2025
```

### Offline sample

`data/sample/sample_matches.csv` is now **fully synthetic**: invented club names, simulated scores and no real bookmaker odds. It exists so CI and the modelling pipeline can run without network access. The earlier portfolio version mixed real-looking 2024/25 fixtures with a description that called the sample invented; v0.4 removes that ambiguity entirely.

Regenerate it with:

```bash
python src/synthetic_data.py --seed 2026 --cycles 2 --rho -0.07
```

## Standings engine

`build_table.py` validates every score/result pair and calculates played, wins, draws, losses, goals for/against, goal difference and points.

```bash
python src/build_table.py data/sample/sample_matches.csv
```

The primary ordering is:

1. points;
2. goal difference;
3. goals scored.

If teams remain exactly equal on all three, they receive the same rank in this project rather than being silently separated alphabetically. The Premier League has additional final-season head-to-head / away-goals / play-off provisions for rare unresolved cases; this small table engine deliberately does not pretend to implement those exceptional rules.

## Poisson model

The original repository used an iterative strength-scaling heuristic and then reported likelihood on the same 15 rows used for fitting. v0.4 replaces that with a maximum-likelihood attack/defence model and a chronological hold-out.

For home team \(i\) and away team \(j\):

\[
\log \lambda_H = \beta_0 + h + a_i + d_j,
\]

\[
\log \lambda_A = \beta_0 + a_j + d_i.
\]

Here `a` is attacking strength and `d` is a **conceding effect** (larger means more goals allowed). Attack and defence effects each sum to zero for identifiability. Goals are independent Poisson under the base model.

A Dixon–Coles parameter \(\rho\) can then reweight only 0–0, 0–1, 1–0 and 1–1. Crucially, \(\rho\) is fitted on the training set only.

```bash
python src/poisson_model.py data/sample/sample_matches.csv --train-fraction 0.5
```

The output includes:

- training and held-out scoreline log loss for independent Poisson and Dixon–Coles;
- held-out 1X2 log loss, multiclass Brier score and accuracy;
- a training-frequency 1X2 baseline;
- a de-margined bookmaker benchmark when suitable odds columns are present;
- fitted home-goal multiplier and Dixon–Coles \(\rho\);
- a warning flag if \(\rho\) lands near its search boundary.

The bundled 112-match synthetic example intentionally does **not** produce a heroic headline: with only 56 fitting matches, its fitted low-score correction improves training fit but worsens held-out scoreline log loss. That is useful evidence about variance, not a failure to hide.

## Known-truth robustness study

`robustness.py` asks a cleaner question: if the truth about low-score dependence is known, when does Dixon–Coles actually generalise?

```bash
python src/robustness.py --worlds 30
```

It repeats four conditions over 30 independent synthetic worlds each:

| Training matches | True \(\rho\) | Median fitted \(\rho\) | Median held-out NLL gain, independent − DC | DC better on held-out |
|---:|---:|---:|---:|---:|
| 56 | 0.00 | -0.055 | **-0.0089** | 26.7% |
| 56 | -0.12 | -0.168 | **-0.0022** | 40.0% |
| 224 | 0.00 | -0.027 | **-0.0018** | 40.0% |
| 224 | -0.12 | -0.137 | **+0.0033** | 63.3% |

Positive gain means Dixon–Coles has lower held-out scoreline log loss.

The important pattern is visible across the whole experiment:

- the extra parameter improves training fit because it is selected there;
- with no true low-score dependence, its median held-out effect is negative at both sample sizes;
- with deliberately planted \(\rho=-0.12\), 56 training matches are still too noisy for reliable benefit;
- with 224 training matches, the fitted median \(\rho=-0.137\) is much closer to the planted value and the median held-out effect becomes positive.

This is a **known-truth simulation about model complexity and information**, not a claim that \(\rho=-0.12\) describes the Premier League.

Generated figures:

- `reports/figures/dc_generalisation.png`
- `reports/figures/rho_recovery.png`

## Why the old `-4.209 vs -4.221` headline is retired

That comparison came from fitting and scoring the Dixon–Coles correction on the same tiny sample. Because \(\rho\) is an extra fitted parameter, a small in-sample likelihood improvement is not evidence that the model predicts new matches better. The new public story is the held-out known-truth experiment above.

## Tests

```bash
pytest -q
```

The suite covers:

- match-schema and FTR/score consistency;
- points, goal difference and shared exact-tie ranks;
- deterministic synthetic data;
- MLE attack-strength recovery on a controlled world;
- normalised score/outcome probabilities;
- Dixon–Coles training likelihood behaviour;
- chronological held-out evaluation;
- log-loss/Brier scoring;
- de-margined odds handling;
- FBref table selection by schema;
- fail-closed season-code parsing.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the statistical details and audit decisions.

## Licence

MIT, see [LICENSE](LICENSE).
