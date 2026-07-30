"""
Turns raw play-by-play events into per-moment game states.

Each output row is a snapshot: at this instant in this game, here is the score, the time
left, who has the ball — and, as the label, whether the home team went on to win.

Three properties of the raw feed drive the design here, all established by inspecting the
data rather than by assuming:

  * `actionNumber` is not reliably chronological. Roughly 4,000 rows a season sit out of
    order, with a median backwards clock jump of 75 seconds and a maximum of a full
    12-minute quarter. Ordering is therefore driven by the game clock, with `actionNumber`
    breaking ties inside the same instant.

  * The running score is present on only ~26% of rows, and administrative rows carry stale
    or corrupt values — a period-end marker recording 119-112 after a buzzer-beater had
    made it 122-112, or a row reading 0-1 in the closing seconds of a 115-113 game. Since
    a basketball score can never decrease, a running maximum repairs both failure modes.
    This reconstructs the correct final score in all 6,150 games.

  * Possession is never stated. It is inferred from the events that establish it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_INTERIM, DATA_RAW, SEASONS
from utils import get_logger, parse_clock

log = get_logger("game_states")

REGULATION_PERIOD_SECONDS = 720  # 12:00
OVERTIME_PERIOD_SECONDS = 300  # 5:00

# Administrative rows: they carry no game state, and are the worst offenders for
# out-of-order timestamps and stale scores.
DROP_ACTION_TYPES = {"Instant Replay"}

# Possession is not stated in the feed, so it is inferred from what each event implies about
# who holds the ball *afterwards*. The direction matters: a made basket and a turnover both
# hand the ball to the OPPONENT of the team credited with the event. Crediting the acting
# team in those cases inverts the feature — with the naive version, home teams tied in the
# last 30 seconds appeared to win more often WITHOUT the ball (58.2%) than with it (55.1%).
POSSESSION_TO_OPPONENT = {"Made Shot", "Turnover"}
POSSESSION_TO_ACTOR = {"Missed Shot", "Rebound"}


def seconds_remaining_in_game(period: pd.Series, clock: pd.Series) -> pd.Series:
    """Seconds left in regulation; inside overtime, seconds left in the current OT period."""
    regulation = (4 - period).clip(lower=0) * REGULATION_PERIOD_SECONDS + clock
    return np.where(period <= 4, regulation, clock)


def build_season(season: str) -> pd.DataFrame:
    pbp = pd.read_parquet(DATA_RAW / f"pbp_{season}.parquet")
    flags = pd.read_parquet(DATA_INTERIM / f"game_flags_{season}.parquet")

    pbp["gameId"] = pbp["gameId"].astype(str)
    pbp = pbp[~pbp["actionType"].isin(DROP_ACTION_TYPES)].copy()

    # --- chronological order: clock first, actionNumber only to break ties ---
    pbp["clock_seconds"] = pbp["clock"].map(parse_clock)
    pbp = pbp.sort_values(
        ["gameId", "period", "clock_seconds", "actionNumber"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    # --- score: forward-fill, then enforce that it never decreases ---
    pbp["score_home"] = pd.to_numeric(pbp["scoreHome"].replace("", np.nan))
    pbp["score_away"] = pd.to_numeric(pbp["scoreAway"].replace("", np.nan))
    grp = pbp.groupby("gameId")[["score_home", "score_away"]]
    pbp[["score_home", "score_away"]] = grp.ffill().fillna(0)
    pbp[["score_home", "score_away"]] = pbp.groupby("gameId")[
        ["score_home", "score_away"]
    ].cummax()

    # --- time ---
    pbp["seconds_remaining"] = seconds_remaining_in_game(pbp["period"], pbp["clock_seconds"])
    pbp["is_overtime"] = pbp["period"] >= 5

    # --- attach game context and the outcome being predicted ---
    # Neutral-site games are excluded: with no true home team, they carry no home-court
    # effect for the model to learn. The Alamodome and Austin's Moody Center are treated as
    # home (see identify_neutral_sites.py), so only genuinely neutral venues drop out here.
    n_neutral = int(flags["neutral_site"].sum())
    ctx = flags[~flags["neutral_site"]][
        [
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "home_team_id",
            "away_team_id",
            "home_win",
            "neutral_site",
            "venue_known",
        ]
    ]
    log.info(f"{season}: excluding {n_neutral} neutral-site game(s)")
    df = pbp.merge(ctx, left_on="gameId", right_on="game_id", how="inner")

    df["score_margin"] = df["score_home"] - df["score_away"]

    # --- possession: who holds the ball after each event ---
    # A steal arrives as a row with a blank actionType whose description reads "... STEAL".
    is_steal = (df["actionType"] == "") & df["description"].str.contains("STEAL", na=False)

    # A free throw surrenders the ball only if it is made AND is the last of its set. The
    # feed records a score on made free throws only, so a populated score means it went in.
    ft_made = (df["actionType"] == "Free Throw") & df["scoreHome"].ne("")
    ft_nums = df["subType"].str.extract(r"(\d+) of (\d+)")
    ft_last = (ft_nums[0] == ft_nums[1]) | df["subType"].str.contains("Technical", na=False)
    ft_surrenders = ft_made & ft_last.fillna(False)

    has_team = df["teamId"].ne(0)
    acting_is_home = df["teamId"] == df["home_team_id"]
    to_opponent = (df["actionType"].isin(POSSESSION_TO_OPPONENT) | ft_surrenders) & has_team
    to_actor = (df["actionType"].isin(POSSESSION_TO_ACTOR) | is_steal) & has_team

    home_gets_ball = np.where(
        to_opponent, ~acting_is_home, np.where(to_actor, acting_is_home, np.nan)
    )
    df["home_has_ball"] = pd.Series(home_gets_ball, index=df.index).astype("boolean")
    df["home_has_ball"] = df.groupby("game_id")["home_has_ball"].ffill()

    out = df[
        [
            "game_id",
            "game_date",
            "home_team",
            "away_team",
            "period",
            "clock_seconds",
            "seconds_remaining",
            "is_overtime",
            "score_home",
            "score_away",
            "score_margin",
            "home_has_ball",
            "actionType",
            "subType",
            "shotValue",
            "teamId",
            "personId",
            "description",
            "neutral_site",
            "venue_known",
            "home_win",
        ]
    ].copy()

    return out.reset_index(drop=True)


def validate(df: pd.DataFrame, season: str) -> None:
    games = pd.read_parquet(DATA_RAW / f"games_{season}.parquet")

    final = df.groupby("game_id")[["score_home", "score_away"]].last()
    m = games.merge(final, left_on="game_id", right_index=True, how="inner")
    ok = (m["home_pts"] == m["score_home"]) & (m["away_pts"] == m["score_away"])
    if not ok.all():
        raise RuntimeError(f"{season}: {(~ok).sum()} game(s) reconstruct to the wrong final score")

    if df["seconds_remaining"].min() < 0:
        raise RuntimeError(f"{season}: negative time remaining")

    # The label must be constant within a game and must agree with the final score.
    per_game = df.groupby("game_id").agg(
        win=("home_win", "nunique"), sh=("score_home", "last"), sa=("score_away", "last"),
        lab=("home_win", "last"),
    )
    if (per_game["win"] != 1).any():
        raise RuntimeError(f"{season}: home_win is not constant within every game")
    if not (per_game["lab"] == (per_game["sh"] > per_game["sa"]).astype(int)).all():
        raise RuntimeError(f"{season}: home_win disagrees with the reconstructed final score")

    unknown_poss = df["home_has_ball"].isna().mean()
    log.info(
        f"{season}: {df['game_id'].nunique()} games, {len(df)} states, "
        f"possession unknown on {unknown_poss*100:.2f}% of rows"
    )


def main():
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        df = build_season(season)
        validate(df, season)
        df.to_parquet(DATA_INTERIM / f"game_states_{season}.parquet", index=False)
        log.info(f"{season}: wrote game_states_{season}.parquet")


if __name__ == "__main__":
    main()
