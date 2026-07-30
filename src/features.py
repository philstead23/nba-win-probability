"""
Builds model features on top of the per-moment game states.

Everything here must be knowable *at that instant in the game*. Nothing may look ahead —
not to the final score, and not to games played later. The pre-game strength feature is the
one with real leakage risk, and is built with an explicit shift so a game never sees itself
or anything after it.

Feature groups:
  * situation   — margin, time, possession, period (carried through from game states)
  * fouls       — team fouls in the current period, and whether the bonus is in effect
  * timeouts    — timeouts used so far by each team
  * momentum    — how the margin has moved over the last few minutes
  * strength    — each team's recent form coming into the game
  * interaction — margin scaled by how much time is left to erase it
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.static import teams as static_teams

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_INTERIM, DATA_PROCESSED, DATA_RAW, SEASONS, SEED_SEASON
from utils import get_logger, parse_clock

log = get_logger("features")

REGULATION_PERIOD_SECONDS = 720
OVERTIME_PERIOD_SECONDS = 300

# Fouls that do NOT count toward the team total. The three non-technical entries —
# offensive, offensive charge, defensive three seconds — were identified from the feed itself:
# the NBA publishes a team-foul number on foul descriptions, and never publishes one for these.
# Including them (an earlier mistake) disagreed with the league's own count on 15% of fouls;
# excluding them gives 100.00% agreement across all 32,951 fouls where the number is published.
NON_TEAM_FOULS = {
    "Offensive",
    "Offensive Charge",
    "Defense 3 Second",
    "Double Personal",
    "Technical",
    "Double Technical",
    "Hanging Technical",
    "Delay Technical",
    "Flopping",
    "Non-Unsportsmanlike Technical",
    "Too Many Players Technical",
    "Excess Timeout Technical",
    "Bench",
}

# Retained for reference and for the foul-count feature. The bonus flag itself is no longer
# derived from this threshold — it is read from the feed's own "PN" (penalty) marker, which
# also captures the last-two-minutes rule that a fixed threshold of 5 misses entirely: 792 of
# 793 disagreements between the two approaches occurred inside the final 2:00 of a period,
# where the 2nd team foul puts a team in the penalty regardless of the running count.
BONUS_THRESHOLD = 5

MOMENTUM_WINDOW_SECONDS = 180  # 3 minutes of game clock

# Timeouts allowed per team: 7 in regulation, plus 2 for each overtime period. Established
# from the data rather than recalled — across 11,680 regulation team-games, 1,885 stop at
# exactly 7 and only 19 (0.16%) exceed it, and the two-overtime maximum of 11 is exactly
# 7 + 2 + 2. The rare excesses are clamped rather than treated as evidence of a higher limit.
TIMEOUTS_PER_GAME = 7
TIMEOUTS_PER_OVERTIME = 2
FORM_WINDOW_GAMES = 15

NICKNAME_TO_ABBR = {t["nickname"].lower(): t["abbreviation"] for t in static_teams.get_teams()}


def elapsed_seconds(period: pd.Series, clock: pd.Series) -> pd.Series:
    """Seconds of game time played so far, strictly increasing across periods and overtimes.

    The count of completed periods must be `min(period - 1, 4)`, not `min(period, 4) - 1`.
    The latter credits an overtime period with only three completed quarters, so the start of
    OT lands at 2160s — exactly the start of the 4th — and overtime rows sort *inside* the 4th
    quarter instead of after it. That silently corrupts row order, momentum, and the
    foul/timeout alignment for every game that goes to overtime.
    """
    completed_regulation = (period - 1).clip(upper=4)
    completed_overtime = (period - 5).clip(lower=0)
    before = (
        completed_regulation * REGULATION_PERIOD_SECONDS
        + completed_overtime * OVERTIME_PERIOD_SECONDS
    )
    period_length = np.where(period <= 4, REGULATION_PERIOD_SECONDS, OVERTIME_PERIOD_SECONDS)
    return before + (period_length - clock)


def add_foul_features(states: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Team-foul count in the current period, and the resulting bonus state.

    The count is computed here, but the rule behind it was **read off the feed rather than
    recalled**. NBA foul descriptions embed the league's own team-foul number — "Phillips
    P.FOUL (P1.T1)" is that player's 1st personal and the team's 1st team foul — which makes
    it possible to check a hand count against the official one instead of trusting a
    remembered rule.

    That check found a real error. An earlier version tallied every non-technical foul and
    disagreed with the NBA's number on 15% of fouls, always running one high after the first
    quarter. The cause: **offensive fouls, offensive charges and defensive three-second
    violations do not count toward the team total.** In this data, 2,603 offensive fouls, 807
    charges and 304 defensive three-seconds carry no team-foul number at all — exactly 0% of
    them — while shooting, personal and loose-ball fouls carry one. Adding those three to the
    exclusion list takes agreement with the league's own numbering to **100.00%** across all
    32,951 fouls where it is published.

    The foul *count* still has to be computed, because the feed's numbering stops at 4 — it is
    never published for a 5th team foul or beyond. But the **bonus flag is read from the feed
    directly**: once a team is in the penalty, its fouls are marked "PN" in place of a number.
    Taking that marker rather than thresholding the count matters, because a fixed threshold
    of 5 misses the last-two-minutes rule. Comparing the two approaches found 793 fouls where
    the NBA said penalty and a count-of-5 rule did not, and **792 of them were inside the final
    2:00 of a period** — where the 2nd team foul puts a team in the penalty. Reading the marker
    captures every such rule without any of them needing to be known.
    """
    fouls = pbp[(pbp["actionType"] == "Foul") & pbp["teamId"].ne(0)].copy()
    counted = fouls[~fouls["subType"].isin(NON_TEAM_FOULS)].copy()
    counted["foul_n"] = counted.groupby(["gameId", "period", "teamId"]).cumcount() + 1

    # The feed marks a foul "PN" instead of numbering it once that team is in the penalty.
    # Reading that marker is better than deriving the penalty from a foul count, because it
    # carries every rule that puts a team in the bonus without any of them having to be known.
    fouls["penalty"] = fouls["description"].str.contains(r"\.PN\)", regex=True, na=False)
    penalty = fouls[fouls["penalty"]][["gameId", "period", "_seq", "teamId"]].copy()
    penalty["in_penalty"] = 1

    out = states
    for side in ("home", "away"):
        side_fouls = counted[["gameId", "period", "_seq", "teamId", "foul_n"]].rename(
            columns={"foul_n": f"{side}_fouls_period"}
        )
        merged = pd.merge_asof(
            out.sort_values("_seq"),
            side_fouls.sort_values("_seq"),
            on="_seq",
            left_by=["game_id", "period", f"{side}_team_id"],
            right_by=["gameId", "period", "teamId"],
            direction="backward",
        )
        out = out.assign(
            **{f"{side}_fouls_period": merged[f"{side}_fouls_period"].fillna(0).values}
        )

        # A team is in the bonus when its OPPONENT is in the penalty, so this looks up the
        # opponent's penalty state and assigns it to this side.
        other = "away" if side == "home" else "home"
        pen = penalty.rename(columns={"in_penalty": f"{side}_in_bonus"})
        merged_pen = pd.merge_asof(
            out.sort_values("_seq"),
            pen.sort_values("_seq"),
            on="_seq",
            left_by=["game_id", "period", f"{other}_team_id"],
            right_by=["gameId", "period", "teamId"],
            direction="backward",
        )
        out = out.assign(
            **{f"{side}_in_bonus": merged_pen[f"{side}_in_bonus"].fillna(0).astype(int).values}
        )
    return out


def add_timeout_features(states: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Timeouts used so far by each team, read from the running count in the description.

    The feed gives no team id on timeout rows (`teamId` is 0 and `teamTricode` is blank), so
    the team comes from the nickname in the description. That mapping resolves every timeout
    row in the data, and never attributes one to a team not playing in that game.
    """
    to = pbp[pbp["actionType"] == "Timeout"].copy()
    to["abbr"] = (
        to["description"].str.split(" Timeout").str[0].str.strip().str.lower().map(NICKNAME_TO_ABBR)
    )
    to["used"] = to["description"].str.extract(r"(?:Full|Reg\.)\s*(\d+)")[0].astype(float)
    to = to.dropna(subset=["abbr", "used"])[["gameId", "_seq", "abbr", "used"]]

    out = states
    for side in ("home", "away"):
        side_to = to.rename(columns={"used": f"{side}_timeouts_used"})
        merged = pd.merge_asof(
            out.sort_values("_seq"),
            side_to.sort_values("_seq"),
            on="_seq",
            left_by=["game_id", f"{side}_team"],
            right_by=["gameId", "abbr"],
            direction="backward",
        )
        out = out.assign(
            **{f"{side}_timeouts_used": merged[f"{side}_timeouts_used"].fillna(0).values}
        )
    return out


def add_momentum(states: pd.DataFrame) -> pd.DataFrame:
    """Change in score margin over the last MOMENTUM_WINDOW_SECONDS of game clock."""
    parts = []
    for _, g in states.groupby("game_id", sort=False):
        g = g.sort_values("elapsed").copy()
        past = pd.merge_asof(
            g[["elapsed"]].assign(target=g["elapsed"] - MOMENTUM_WINDOW_SECONDS),
            g[["elapsed", "score_margin"]].rename(columns={"score_margin": "margin_then"}),
            left_on="target",
            right_on="elapsed",
            direction="backward",
        )
        g["margin_then"] = past["margin_then"].fillna(0).values
        g["momentum"] = g["score_margin"] - g["margin_then"]
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def build_team_form(seasons) -> pd.DataFrame:
    """Each team's average point differential over its previous FORM_WINDOW_GAMES games.

    Strictly backward-looking: the rolling mean is shifted by one game so a team's form
    entering a game never includes that game or any later one.

    The window runs **across the season boundary**, seeded by one extra prior season of final
    scores. Resetting at each September left roughly the first month of every season with no
    form at all — a team needs 15 games before the window fills — so the model went into early
    November games knowing nothing about either team and defaulting to a coin flip decided by
    home court. That was visible in a 2025-11-04 game where it gave the home side 52.7% against
    a much stronger opponent the betting market had favoured by 8.5 points.

    Carrying the prior season forward is imperfect, since rosters change over a summer, but it
    is strictly more information than nothing and it decays out within 15 games. Its value was
    confirmed on the validation split rather than assumed.
    """
    rows = []
    for season in [SEED_SEASON] + list(seasons):
        path = DATA_RAW / f"games_{season}.parquet"
        if not path.exists():
            log.warning(f"{season}: no game list, form cannot be seeded from it")
            continue
        g = pd.read_parquet(path)
        for _, r in g.iterrows():
            rows.append((season, r["game_date"], r["game_id"], r["home_team"], r["home_pts"] - r["away_pts"]))
            rows.append((season, r["game_date"], r["game_id"], r["away_team"], r["away_pts"] - r["home_pts"]))
    tg = pd.DataFrame(rows, columns=["season", "game_date", "game_id", "team", "point_diff"])

    # Sorted by team and date only — deliberately NOT grouped by season, so the window spans
    # the offseason and a team's first games of a year inherit the end of its previous one.
    tg = tg.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)
    tg["form"] = tg.groupby("team", sort=False)["point_diff"].transform(
        lambda s: s.shift(1).rolling(FORM_WINDOW_GAMES, min_periods=FORM_WINDOW_GAMES).mean()
    )
    return tg[["game_id", "team", "form"]]


def build_season(season: str, form: pd.DataFrame) -> pd.DataFrame:
    states = pd.read_parquet(DATA_INTERIM / f"game_states_{season}.parquet")
    pbp = pd.read_parquet(DATA_RAW / f"pbp_{season}.parquet")
    pbp = pbp[pbp["actionType"] != "Instant Replay"].copy()
    pbp["gameId"] = pbp["gameId"].astype(str)
    pbp["clock_seconds"] = pbp["clock"].map(parse_clock)
    pbp = pbp.sort_values(
        ["gameId", "period", "clock_seconds", "actionNumber"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    ids = pd.read_parquet(DATA_INTERIM / f"game_flags_{season}.parquet")[
        ["game_id", "home_team_id", "away_team_id"]
    ]
    states = states.merge(ids, on="game_id", how="left")

    # A single monotonically increasing sequence per season lets merge_asof align events to
    # states without relying on the unreliable actionNumber.
    states["elapsed"] = elapsed_seconds(states["period"], states["clock_seconds"])
    pbp["elapsed"] = elapsed_seconds(pbp["period"], pbp["clock_seconds"])
    states = states.sort_values(["game_id", "elapsed"]).reset_index(drop=True)
    states["_seq"] = np.arange(len(states), dtype=float)

    key = states[["game_id", "elapsed", "_seq"]].rename(columns={"game_id": "gameId"})
    pbp = pd.merge_asof(
        pbp.sort_values("elapsed"),
        key.sort_values("elapsed"),
        on="elapsed",
        by="gameId",
        direction="forward",
    )
    pbp = pbp.dropna(subset=["_seq"])

    states = add_foul_features(states, pbp)
    states = add_timeout_features(states, pbp)
    states = add_momentum(states)

    home_form = form.rename(columns={"team": "home_team", "form": "home_form"})
    away_form = form.rename(columns={"team": "away_team", "form": "away_form"})
    states = states.merge(home_form, on=["game_id", "home_team"], how="left")
    states = states.merge(away_form, on=["game_id", "away_team"], how="left")
    states["form_diff"] = states["home_form"] - states["away_form"]

    # A lead is worth more the less time remains to erase it.
    states["margin_per_minute_left"] = states["score_margin"] / (
        states["seconds_remaining"] / 60.0 + 1.0
    )

    # "Timeouts left" rather than "timeouts used" — that is the unit a coach actually thinks
    # in during a huddle, and it stays meaningful in overtime, where the allowance changes and
    # a raw count of what has been spent does not tell you what remains.
    allowance = TIMEOUTS_PER_GAME + (states["period"] - 4).clip(lower=0) * TIMEOUTS_PER_OVERTIME
    for side in ("home", "away"):
        states[f"{side}_timeouts_left"] = (
            (allowance - states[f"{side}_timeouts_used"]).clip(lower=0)
        )

    # Elo: opponent-adjusted team strength, which trailing point differential is blind to.
    elo_path = DATA_RAW / "elo.parquet"
    if elo_path.exists():
        elo = pd.read_parquet(elo_path)[
            ["game_id", "home_elo", "away_elo", "elo_diff", "elo_home_win_prob"]
        ]
        states = states.merge(elo, on="game_id", how="left")
    else:
        log.warning("elo.parquet not found — run src/elo.py first")

    states["season"] = season
    return states


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    form = build_team_form(SEASONS)
    for season in SEASONS:
        df = build_season(season, form)
        df.to_parquet(DATA_PROCESSED / f"features_{season}.parquet", index=False)
        log.info(
            f"{season}: {len(df):,} rows, {df['game_id'].nunique()} games, "
            f"form known on {df['form_diff'].notna().mean()*100:.1f}% of rows"
        )


if __name__ == "__main__":
    main()
