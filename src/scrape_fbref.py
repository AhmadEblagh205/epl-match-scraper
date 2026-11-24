"""Optional FBref fixture-table scraper.

Prefer a published CSV when one exists.  This route is isolated because HTML
scrapers are inherently more brittle and must be run in accordance with the
site's current robots.txt and terms.
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
URL = ("https://fbref.com/en/comps/9/{season}/schedule/"
       "{season}-Premier-League-Scores-and-Fixtures")
HEADERS = {"User-Agent": "epl-match-scraper/1.0 (personal portfolio project)"}


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[-1]) for c in out.columns]
    else:
        out.columns = [str(c) for c in out.columns]
    return out


def parse_fixture_html(html: str) -> pd.DataFrame:
    tables = [_flatten_columns(t) for t in pd.read_html(io.StringIO(html))]
    candidates = [t for t in tables if {"Date", "Home", "Away", "Score"} <= set(t.columns)]
    if not candidates:
        raise ValueError("no table with Date/Home/Away/Score columns found")
    sched = max(candidates, key=len).dropna(subset=["Score"]).copy()
    goals = sched["Score"].astype(str).str.extract(
        r"(?P<FTHG>\d+)\D+(?P<FTAG>\d+)").apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame({
        "Date": sched["Date"],
        "HomeTeam": sched["Home"],
        "AwayTeam": sched["Away"],
        "FTHG": goals["FTHG"],
        "FTAG": goals["FTAG"],
    }).dropna()
    out[["FTHG", "FTAG"]] = out[["FTHG", "FTAG"]].astype(int)
    out["FTR"] = ["H" if h > a else "A" if a > h else "D"
                  for h, a in zip(out.FTHG, out.FTAG)]
    return out.reset_index(drop=True)


def fetch(season: str, delay: float = 4.0) -> pd.DataFrame:
    url = URL.format(season=season)
    print(f"GET {url}")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    out = parse_fixture_html(resp.text)
    time.sleep(max(float(delay), 0.0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2024-2025")
    ap.add_argument("--delay", type=float, default=4.0)
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    df = fetch(args.season, args.delay)
    dest = RAW / f"fbref_{args.season}.csv"
    df.to_csv(dest, index=False)
    print(f"{len(df)} completed matches -> {dest}")


if __name__ == "__main__":
    main()
