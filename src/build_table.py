"""Build Premier-League-style standings from a results CSV.

The primary in-season ordering is points, goal difference and goals scored.
If teams are still exactly tied on all three, this module gives them the same
rank rather than inventing a hidden alphabetical tiebreak.  Final-season
head-to-head/play-off rules are a separate competition rule and are not
silently approximated here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_validation import validate_matches

ROOT = Path(__file__).resolve().parents[1]


def league_table(matches: pd.DataFrame) -> pd.DataFrame:
    m = validate_matches(matches)
    teams = sorted(set(m.HomeTeam) | set(m.AwayTeam))
    rows = {t: {"P": 0, "W": 0, "D": 0, "L": 0,
                "GF": 0, "GA": 0, "Pts": 0} for t in teams}

    for r in m.itertuples(index=False):
        home, away = rows[r.HomeTeam], rows[r.AwayTeam]
        hg, ag = int(r.FTHG), int(r.FTAG)
        home["P"] += 1; away["P"] += 1
        home["GF"] += hg; home["GA"] += ag
        away["GF"] += ag; away["GA"] += hg
        if hg > ag:
            home["W"] += 1; away["L"] += 1; home["Pts"] += 3
        elif ag > hg:
            away["W"] += 1; home["L"] += 1; away["Pts"] += 3
        else:
            home["D"] += 1; away["D"] += 1
            home["Pts"] += 1; away["Pts"] += 1

    table = pd.DataFrame(rows).T
    table["GD"] = table["GF"] - table["GA"]
    # Alphabetical ordering is only a deterministic display order inside an
    # unresolved exact tie; Rank below remains shared for that tie.
    table["Team"] = table.index
    table = table.sort_values(["Pts", "GD", "GF", "Team"],
                              ascending=[False, False, False, True],
                              kind="stable")
    keys = list(zip(table["Pts"], table["GD"], table["GF"]))
    ranks, last_key, rank = [], None, 0
    for pos, key in enumerate(keys, start=1):
        if key != last_key:
            rank = pos
            last_key = key
        ranks.append(rank)
    table.insert(0, "Rank", ranks)
    table = table.drop(columns="Team")
    table.index.name = "Team"
    return table[["Rank", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="results CSV")
    ap.add_argument("--top", type=int, default=None)
    args = ap.parse_args()
    matches = pd.read_csv(args.csv)
    table = league_table(matches)
    if args.top:
        table = table.head(args.top)
    print(table.to_string())
    out = ROOT / "reports" / "league_table.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
