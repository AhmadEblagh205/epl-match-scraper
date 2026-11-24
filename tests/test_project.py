import numpy as np
import pandas as pd
import pytest

from build_table import league_table
from data_validation import validate_matches
from download_football_data import validate_season_code
from poisson_model import (
    chronological_split,
    evaluate_chronological,
    expected_goals,
    fit_dixon_coles_rho,
    fit_independent_poisson,
    market_metrics,
    match_probs,
    outcome_metrics,
    score_grid,
    scoreline_log_loss,
)
from scrape_fbref import parse_fixture_html
from synthetic_data import generate_matches


def tiny_matches():
    return pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "HomeTeam": ["A", "B", "A"],
        "AwayTeam": ["B", "C", "C"],
        "FTHG": [2, 1, 0],
        "FTAG": [0, 1, 0],
        "FTR": ["H", "D", "D"],
    })


def test_points_goal_difference_and_shared_exact_tie_rank():
    table = league_table(tiny_matches())
    assert table.loc["A", "Pts"] == 4 and table.loc["A", "GD"] == 2
    assert table.loc["B", "Pts"] == 1
    assert table.loc["C", "Pts"] == 2
    assert list(table.index)[0] == "A"

    tied = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02"],
        "HomeTeam": ["A", "C"], "AwayTeam": ["B", "D"],
        "FTHG": [1, 1], "FTAG": [0, 0], "FTR": ["H", "H"],
    })
    tt = league_table(tied)
    assert tt.loc["A", "Rank"] == tt.loc["C", "Rank"] == 1


def test_validation_rejects_result_score_disagreement():
    m = tiny_matches()
    m.loc[0, "FTR"] = "A"
    with pytest.raises(ValueError, match="disagrees"):
        validate_matches(m)


def test_synthetic_generator_is_reproducible_and_schema_valid():
    a, truth_a = generate_matches(seed=7, cycles=1, rho=-0.05)
    b, truth_b = generate_matches(seed=7, cycles=1, rho=-0.05)
    pd.testing.assert_frame_equal(a, b)
    assert truth_a == truth_b
    assert len(a) == 56
    assert len(set(a.HomeTeam) | set(a.AwayTeam)) == 8
    validate_matches(a)


def test_poisson_fit_recovers_strong_attack_ordering_and_probabilities():
    m, truth = generate_matches(seed=12, cycles=5, rho=0.0)
    fit = fit_independent_poisson(m)
    best_true = max(truth.attack, key=truth.attack.get)
    worst_true = min(truth.attack, key=truth.attack.get)
    assert fit.attack[best_true] > fit.attack[worst_true]
    mh, ma = expected_goals(fit, best_true, worst_true)
    h, d, a = match_probs(mh, ma)
    assert np.isclose(h + d + a, 1.0)
    assert h > a


def test_score_grid_is_nonnegative_and_normalised():
    grid = score_grid(1.5, 1.1, rho=-0.08)
    assert np.all(grid >= 0)
    assert np.isclose(grid.sum(), 1.0)


def test_dixon_coles_training_fit_cannot_be_worse_after_rho_fit():
    m, _ = generate_matches(seed=4, cycles=3, rho=-0.08)
    train, _ = chronological_split(m, 0.5)
    fit = fit_independent_poisson(train)
    independent = scoreline_log_loss(train, fit, rho=0.0)
    fit_dixon_coles_rho(train, fit)
    dc = scoreline_log_loss(train, fit, rho=fit.rho)
    assert dc <= independent + 1e-8


def test_chronological_evaluation_uses_heldout_matches():
    m, _ = generate_matches(seed=11, cycles=4, rho=-0.08)
    result = evaluate_chronological(m, 0.5)
    assert result["n_train"] > 0 and result["n_test"] > 0
    assert result["n_train"] + result["n_test"] == len(m)
    assert "test_outcome_dc" in result
    assert result["test_outcome_dc"]["log_loss"] > 0


def test_outcome_metrics_reward_better_probabilities():
    matches = pd.DataFrame({"FTR": ["H", "D", "A"]})
    good = pd.DataFrame({
        "p_home": [0.8, 0.1, 0.1], "p_draw": [0.1, 0.8, 0.1],
        "p_away": [0.1, 0.1, 0.8],
    })
    bad = pd.DataFrame({
        "p_home": [1/3]*3, "p_draw": [1/3]*3, "p_away": [1/3]*3,
    })
    assert outcome_metrics(good, matches)["log_loss"] < outcome_metrics(bad, matches)["log_loss"]
    assert outcome_metrics(good, matches)["brier"] < outcome_metrics(bad, matches)["brier"]


def test_market_probabilities_are_demargined_before_scoring():
    m = pd.DataFrame({
        "FTR": ["H", "D"],
        "B365H": [2.0, 2.0], "B365D": [4.0, 4.0], "B365A": [4.0, 4.0],
    })
    metrics, label = market_metrics(m)
    assert "Bet365" in label
    assert np.isfinite(metrics["log_loss"])
    assert 0 <= metrics["brier"] <= 2


def test_scraper_parser_selects_by_required_columns_not_largest_table():
    html = """
    <table><tr><th>Noise</th></tr>""" + "".join("<tr><td>x</td></tr>" for _ in range(10)) + """</table>
    <table>
      <tr><th>Date</th><th>Home</th><th>Away</th><th>Score</th></tr>
      <tr><td>2024-08-01</td><td>A</td><td>B</td><td>2–1</td></tr>
      <tr><td>2024-08-02</td><td>C</td><td>D</td><td>0-0</td></tr>
    </table>
    """
    out = parse_fixture_html(html)
    assert list(out.FTR) == ["H", "D"]
    assert list(out.FTHG) == [2, 0]


def test_scraper_parser_fails_if_fixture_table_absent():
    with pytest.raises(ValueError, match="no table"):
        parse_fixture_html("<table><tr><th>Other</th></tr><tr><td>x</td></tr></table>")


def test_season_code_validation_is_fail_closed():
    assert validate_season_code("2425") == "2425"
    with pytest.raises(ValueError):
        validate_season_code("24-25")
    with pytest.raises(ValueError):
        validate_season_code("2426")
