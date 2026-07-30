"""
Elo ratings for NBA teams, used as the pre-game team strength feature.

This replaces a trailing 15-game point differential, which weights every opponent equally:
+6 per game compiled against a soft slate counts the same as +6 against contenders. Tip-off
estimates from that feature sat near a coin flip regardless of who was playing.

Elo prices each result against the opponent's rating instead, so the same win is worth more
against a strong team than a weak one and schedule difficulty is absorbed by the rating rather
than left out of it.

Design choices, all standard for basketball Elo:
  * **Margin of victory matters**, with diminishing returns — a 30-point win is worth more than
    a 3-point win, but not ten times more.
  * **Autocorrelation correction.** Good teams run up bigger margins partly *because* they are
    already rated highly, which would otherwise inflate them without limit. The margin
    multiplier is damped by the rating gap.
  * **Between seasons, ratings regress toward the mean.** Rosters change over a summer, so
    carrying a rating forward untouched overstates how much last year tells you about this one.

No leakage is possible by construction: a game's rating is the rating *before* it is played,
and the update happens only after the result is recorded.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_RAW, SEASONS, SEED_SEASON
from utils import get_logger

log = get_logger("elo")

START_RATING = 1500.0
K_FACTOR = 20.0
# Home advantage in Elo points, calibrated from these five seasons rather than taken from a
# reference. The commonly quoted figure of 100 comes from an era when home court was worth far
# more; 100 implies a 64% home win rate, and the actual rate here is 55.3%, which implies 37.
HOME_ADVANTAGE = 37.0

# Points of margin per Elo point, fitted on the same data: 1 point of spread ~ 24 Elo. Used
# only to translate between a coach-facing point spread and a rating gap in the dashboard.
ELO_PER_POINT_OF_SPREAD = 24.0
SEASON_REGRESSION = 0.25  # fraction pulled back toward the mean each offseason


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def margin_multiplier(margin: int, rating_diff_winner: float) -> float:
    """Diminishing credit for blowouts, damped by how favoured the winner already was."""
    return (abs(margin) + 3.0) ** 0.8 / (7.5 + 0.006 * rating_diff_winner)


def build_elo() -> pd.DataFrame:
    """One row per game with each side's rating BEFORE that game was played."""
    frames = []
    for season in [SEED_SEASON] + list(SEASONS):
        path = DATA_RAW / f"games_{season}.parquet"
        if not path.exists():
            log.warning(f"{season}: no game list; Elo cannot include it")
            continue
        g = pd.read_parquet(path)[
            ["game_id", "game_date", "home_team", "away_team", "home_pts", "away_pts"]
        ].copy()
        g["season"] = season
        frames.append(g)

    games = pd.concat(frames, ignore_index=True)
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    ratings = {}
    last_season = None
    rows = []

    for r in games.itertuples(index=False):
        if r.season != last_season and last_season is not None:
            # Offseason: pull every rating part-way back toward average.
            for team in ratings:
                ratings[team] = (
                    START_RATING * SEASON_REGRESSION + ratings[team] * (1 - SEASON_REGRESSION)
                )
        last_season = r.season

        home = ratings.setdefault(r.home_team, START_RATING)
        away = ratings.setdefault(r.away_team, START_RATING)

        rows.append(
            {
                "game_id": r.game_id,
                "season": r.season,
                "home_elo": home,
                "away_elo": away,
                "elo_diff": home - away,
                # The rating's own view of the game, home court included. Useful as a feature
                # in its own right and as a sanity check against the market.
                "elo_home_win_prob": expected_score(home + HOME_ADVANTAGE, away),
            }
        )

        margin = r.home_pts - r.away_pts
        home_won = margin > 0
        actual = 1.0 if home_won else 0.0
        expected = expected_score(home + HOME_ADVANTAGE, away)

        # Rating edge held by whoever actually won, used to damp blowout credit.
        winner_edge = (home + HOME_ADVANTAGE - away) if home_won else (away - home - HOME_ADVANTAGE)
        shift = K_FACTOR * margin_multiplier(margin, winner_edge) * (actual - expected)

        ratings[r.home_team] = home + shift
        ratings[r.away_team] = away - shift

    out = pd.DataFrame(rows)
    log.info(
        f"built Elo for {len(out)} games across {out['season'].nunique()} seasons "
        f"(range {out['home_elo'].min():.0f}-{out['home_elo'].max():.0f})"
    )
    return out


if __name__ == "__main__":
    elo = build_elo()
    elo.to_parquet(DATA_RAW / "elo.parquet", index=False)

    # How well does Elo alone call games, before any in-game information?
    games = pd.concat(
        [pd.read_parquet(DATA_RAW / f"games_{s}.parquet").assign(season=s) for s in SEASONS]
    )
    m = games.merge(elo, on="game_id")
    from sklearn.metrics import log_loss, roc_auc_score

    print(f"\nElo alone, pre-game, {len(m)} games:")
    print(f"  accuracy picking the winner : {((m.elo_home_win_prob > 0.5) == (m.home_win == 1)).mean()*100:.1f}%")
    print(f"  log loss                    : {log_loss(m.home_win, m.elo_home_win_prob):.4f}")
    print(f"  AUC                         : {roc_auc_score(m.home_win, m.elo_home_win_prob):.4f}")
    print(f"  (a constant 55.3% would give a log loss of {log_loss(m.home_win, np.full(len(m), 0.553)):.4f})")
