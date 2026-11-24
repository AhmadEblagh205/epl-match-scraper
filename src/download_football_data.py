"""Download Premier League result CSVs from football-data.co.uk.

The network path is intentionally small and validated: season-code format,
HTTP status, non-empty CSV and the core result schema are checked before a
file is accepted.
"""
from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import pandas as pd
import requests

from data_validation import REQUIRED_MATCH_COLUMNS

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
BASE = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
HEADERS = {"User-Agent": "epl-match-scraper/1.0 (personal portfolio project)"}


def validate_season_code(season: str) -> str:
    if not re.fullmatch(r"\d{4}", season):
        raise ValueError("season code must look like 2425 for 2024/25")
    if int(season[2:]) != (int(season[:2]) + 1) % 100:
        raise ValueError("season code years must be consecutive, e.g. 2425")
    return season


def download_season(season: str, *, timeout: float = 30.0) -> pd.DataFrame:
    season = validate_season_code(season)
    url = BASE.format(season=season)
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    if not resp.content.strip():
        raise ValueError("download returned an empty file")
    df = pd.read_csv(io.BytesIO(resp.content))
    missing = [c for c in REQUIRED_MATCH_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"downloaded CSV missing core columns: {missing}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=["2425"],
                    help="season codes like 2324 for 2023/24")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for season in args.seasons:
        season = validate_season_code(season)
        dest = RAW / f"epl_{season}.csv"
        if dest.exists():
            print(f"Already downloaded: {dest}")
            continue
        print(f"Downloading {BASE.format(season=season)}")
        df = download_season(season)
        df.to_csv(dest, index=False)
        print(f"{len(df)} rows -> {dest}")


if __name__ == "__main__":
    main()
