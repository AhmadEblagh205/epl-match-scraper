"""Shared validation helpers for match-result inputs."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_MATCH_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")
VALID_RESULTS = {"H", "D", "A"}


def validate_matches(matches: pd.DataFrame, *, require_date: bool = True) -> pd.DataFrame:
    """Return a clean copy after checking the core match schema.

    Validation is deliberately fail-closed.  A bad score/result row should not
    silently enter the league table or the statistical model.
    """
    required = list(REQUIRED_MATCH_COLUMNS if require_date else REQUIRED_MATCH_COLUMNS[1:])
    missing = [c for c in required if c not in matches.columns]
    if missing:
        raise ValueError(f"missing required match columns: {missing}")
    if matches.empty:
        raise ValueError("match data are empty")

    out = matches.copy()
    if require_date:
        parsed = pd.to_datetime(out["Date"], errors="coerce", format="mixed", dayfirst=True)
        if parsed.isna().any():
            bad = out.loc[parsed.isna(), "Date"].astype(str).head(3).tolist()
            raise ValueError(f"unparseable Date values, examples: {bad}")
        out["Date"] = parsed

    for col in ("HomeTeam", "AwayTeam"):
        if out[col].isna().any() or (out[col].astype(str).str.strip() == "").any():
            raise ValueError(f"{col} contains blank team names")
        out[col] = out[col].astype(str).str.strip()

    if (out["HomeTeam"] == out["AwayTeam"]).any():
        raise ValueError("a team cannot play itself")

    for col in ("FTHG", "FTAG"):
        vals = pd.to_numeric(out[col], errors="coerce")
        if vals.isna().any() or (vals < 0).any() or (~np.isclose(vals, np.round(vals))).any():
            raise ValueError(f"{col} must contain non-negative integer goals")
        out[col] = vals.astype(int)

    out["FTR"] = out["FTR"].astype(str).str.upper().str.strip()
    if not set(out["FTR"]).issubset(VALID_RESULTS):
        raise ValueError("FTR must contain only H, D or A")

    implied = np.where(out["FTHG"] > out["FTAG"], "H",
                       np.where(out["FTHG"] < out["FTAG"], "A", "D"))
    bad = out["FTR"].to_numpy() != implied
    if bad.any():
        idx = list(out.index[bad][:3])
        raise ValueError(f"FTR disagrees with full-time score at rows {idx}")

    return out


def ensure_probability_rows(values: np.ndarray, *, atol: float = 1e-8) -> None:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("outcome probabilities must have shape (n, 3)")
    if not np.isfinite(arr).all() or (arr < 0).any() or (arr > 1).any():
        raise ValueError("outcome probabilities must be finite and in [0, 1]")
    if not np.allclose(arr.sum(axis=1), 1.0, atol=atol):
        raise ValueError("outcome probability rows must sum to one")
