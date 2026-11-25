# Premier League Data & Poisson Evaluation Laboratory

[![CI](https://github.com/AhmadEblagh205/epl-match-scraper/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/epl-match-scraper/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

I built this to collect completed Premier League results, validate them into one schema, build standings and fit an attack/defence Poisson model that is scored on matches it was not fitted on. Results come from published CSV files on football-data.co.uk, or from an optional FBref HTML scraper. The sample in `data/sample/` is fully synthetic, with invented clubs and simulated scores, so the tests and models run without a network.

## What it does

- downloads Premier League season CSVs and checks the season code, response and core columns before saving;
- scrapes an FBref results table as a less stable second route;
- builds a standings table from points, goal difference and goals scored;
- fits a maximum-likelihood Poisson model with an optional Dixon-Coles low-score correction, using a chronological hold-out;
- compares the model with a training-frequency baseline and, where odds exist, a bookmaker benchmark with the margin removed.

## Results

On the bundled 112-match synthetic sample, with 56 fitting matches, the Dixon-Coles correction improves training scoreline log loss but worsens the held-out figure (3.2202 against 3.1988 for independent Poisson).

To see when the correction helps, `robustness.py` repeats four conditions over 30 simulated worlds each. A positive gain means Dixon-Coles has lower held-out scoreline log loss.

| Training matches | True rho | Median fitted rho | Median held-out gain | DC better on held-out |
|---:|---:|---:|---:|---:|
| 56 | 0.00 | -0.055 | -0.0089 | 26.7% |
| 56 | -0.12 | -0.168 | -0.0022 | 40.0% |
| 224 | 0.00 | -0.027 | -0.0018 | 40.0% |
| 224 | -0.12 | -0.137 | +0.0033 | 63.3% |

The correction only helps when the dependence is real and there are enough matches to estimate it. This is a simulation and does not describe the Premier League. Figures are in `reports/figures/` (`dc_generalisation.png`, `rho_recovery.png`).

## How to run

```bash
pip install -r requirements.txt
python src/download_football_data.py --seasons 2324 2425
python src/scrape_fbref.py --season 2024-2025
python src/build_table.py data/sample/sample_matches.csv
python src/poisson_model.py data/sample/sample_matches.csv --train-fraction 0.5
python src/robustness.py --worlds 30
python src/synthetic_data.py --seed 2026 --cycles 2 --rho -0.07
pytest -q
```

`synthetic_data.py` regenerates the synthetic sample.

## Notes

Prefer the CSV route. If you use the FBref scraper, check FBref's current robots.txt and terms first. Teams level on points, goal difference and goals scored share a rank, because the table does not implement the Premier League's extra tie-break rules. I retired an older headline comparison (-4.209 against -4.221) because it was fitted and scored on the same small sample. The odds handling, model equations, the sample history and the audit decisions are in docs/METHODOLOGY.md.

## Licence

MIT, see [LICENSE](LICENSE).
