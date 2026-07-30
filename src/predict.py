"""
The single prediction path used by everything outside training.

This module exists to remove a specific failure mode. If the dashboard assembled its own
feature vector, any drift from how training built them — a different default, a formula
applied in a different order, a column in the wrong position — would produce confident,
plausible, wrong numbers with no error raised. So feature construction lives here once, and
`tests/test_predict.py` checks that a state rebuilt through this path reproduces the pipeline's
own stored features and predictions exactly.
"""

import json
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR
from features import TIMEOUTS_PER_GAME
from elo import ELO_PER_POINT_OF_SPREAD, HOME_ADVANTAGE, expected_score
from train_model import FEATURES, PRIMARY_MODEL, to_matrix

REGULATION_PERIOD_SECONDS = 720
OVERTIME_PERIOD_SECONDS = 300

# Defaults that carry no information about who is winning: an evenly matched pair of teams
# with no recent run either way.
NEUTRAL_DEFAULTS = {
    "momentum": 0.0,
    "elo_diff": 0.0,
}

# Timeout and foul state, by contrast, must depend on WHEN in the game you are. A flat
# "two timeouts used" is right early and badly wrong with two minutes left in the fourth,
# where the median is five — enough to move a tied-game reading by more than ten points.
# These medians are computed from the training seasons by train_model.context_defaults.
CONTEXTUAL_FIELDS = (
    "home_timeouts_left",
    "away_timeouts_left",
    "home_fouls_period",
    "away_fouls_period",
    "home_in_bonus",
    "away_in_bonus",
)


@lru_cache(maxsize=1)
def _context_table() -> dict:
    with open(MODELS_DIR / "context_defaults.json") as f:
        return json.load(f)


def contextual_defaults(period, clock_seconds) -> pd.DataFrame:
    """Median timeout and foul state at this point of this period, from the training data.

    Keyed on the clock WITHIN the period, matching train_model.context_defaults. Keying on
    total time left in the game collapsed all of the first quarter into one bucket, so a
    tip-off lookup returned mid-quarter averages instead of the zeros that are actually true.
    """
    table = _context_table()
    period = np.atleast_1d(period)
    clock_seconds = np.atleast_1d(clock_seconds)
    keys = [
        f"{int(min(p, 5))}_{int(min(c // 60, 12))}" for p, c in zip(period, clock_seconds)
    ]
    rows = [table["buckets"].get(k, table["overall"]) for k in keys]
    return pd.DataFrame(rows, columns=list(CONTEXTUAL_FIELDS))


@lru_cache(maxsize=1)
def _margin_bounds() -> dict:
    with open(MODELS_DIR / "margin_bounds.json") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _timeout_bounds() -> dict:
    with open(MODELS_DIR / "timeout_bounds.json") as f:
        return json.load(f)


def _elapsed_at(period: int, clock_seconds: float) -> float:
    completed_regulation = min(max(period - 1, 0), 4)
    completed_overtime = max(period - 5, 0)
    period_length = REGULATION_PERIOD_SECONDS if period <= 4 else OVERTIME_PERIOD_SECONDS
    return (
        completed_regulation * REGULATION_PERIOD_SECONDS
        + completed_overtime * OVERTIME_PERIOD_SECONDS
        + (period_length - clock_seconds)
    )


def _timeout_bound(period: int, clock_seconds: float, which: str, default: int) -> int:
    elapsed = _elapsed_at(period, clock_seconds)
    if elapsed <= 0:
        # Nothing played: both teams hold all of them, so floor and ceiling are the same.
        return TIMEOUTS_PER_GAME

    table = _timeout_bounds()[which]
    bucket = int(elapsed // 30)
    while bucket >= 0:
        if str(bucket) in table:
            return int(table[str(bucket)])
        bucket -= 1
    return default


def min_timeouts_at(period: int, clock_seconds: float) -> int:
    """Fewest timeouts a team has actually had left by this point of a game.

    The companion to max_margin_at, and it exists for a bug of exactly the same shape. The
    calculator let a user set one timeout left two minutes into the first quarter and answered
    45.1% — a 12-point penalty — when no team in five seasons has held fewer than six at that
    point. A logistic regression does not look for comparable games; it evaluates its function
    wherever you point it, so the number came from fourth-quarter arithmetic rather than from
    evidence. Bounding the input keeps the tool from inviting the question.
    """
    return _timeout_bound(period, clock_seconds, "floor", 0)


def max_timeouts_at(period: int, clock_seconds: float) -> int:
    """Most timeouts a team has actually had left by this point of a game.

    The mirror of the floor, and a weaker constraint: there is no rule forcing a team to spend
    timeouts, so seven remains reachable deep into a game. It has happened once in 6,135 games —
    Minnesota carried all seven into the fourth at Utah on 2021-12-31 — and this bound keeps
    that possible while ruling out the last six minutes, where nobody has ever held seven.

    The real constraint late is a different one: a team may only *use* four timeouts in the
    fourth quarter, which the data confirms exactly (909 teams used four, none used five). So
    holding seven at the two-minute mark would mean three you could not spend.
    """
    return _timeout_bound(period, clock_seconds, "ceiling", TIMEOUTS_PER_GAME)


def max_margin_at(period: int, clock_seconds: float) -> int:
    """Largest score margin basketball has actually produced by this point of a game.

    Used to bound the calculator's input. Without it a user can describe a 30-point lead at
    12:00 of the first quarter — before a possession has been played — and get a confident
    answer to a state that cannot exist.
    """
    completed_regulation = min(max(period - 1, 0), 4)
    completed_overtime = max(period - 5, 0)
    period_length = REGULATION_PERIOD_SECONDS if period <= 4 else OVERTIME_PERIOD_SECONDS
    elapsed = (
        completed_regulation * REGULATION_PERIOD_SECONDS
        + completed_overtime * OVERTIME_PERIOD_SECONDS
        + (period_length - clock_seconds)
    )
    # Before a single second has been played the score is 0-0 by definition. The 30-second
    # buckets cannot express that on their own — bucket 0 spans the first half-minute and so
    # carries the largest margin reached within it — so tip-off is handled exactly.
    if elapsed <= 0:
        return 0

    table = _margin_bounds()
    bucket = int(elapsed // 30)
    while bucket >= 0:
        if str(bucket) in table:
            return int(table[str(bucket)])
        bucket -= 1
    return 0


@lru_cache(maxsize=None)
def load_model(which: str = PRIMARY_MODEL):
    path = MODELS_DIR / ("logistic.pkl" if which == "logistic" else "lgbm.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def seconds_remaining_from_clock(period: int, clock_seconds: float) -> float:
    """Match build_game_states: time left in regulation, or within the current OT period."""
    if period <= 4:
        return (4 - period) * REGULATION_PERIOD_SECONDS + clock_seconds
    return clock_seconds


def make_states(**kwargs) -> pd.DataFrame:
    """Build a frame of game states from scalars or arrays, filling anything unspecified.

    Requires `score_margin`, `seconds_remaining` and `period`. `margin_per_minute_left` is
    recomputed here with the identical formula used in features.py; it is deliberately not
    accepted as an input, so it can never be passed in inconsistent with the margin and time
    it is derived from.
    """
    for required in ("score_margin", "seconds_remaining", "period"):
        if required not in kwargs:
            raise ValueError(f"{required} is required")

    # Reject anything that is not a real feature. Silently ignoring an unknown name is the
    # exact failure this module exists to prevent: a renamed feature left a caller passing
    # `form_diff=10` long after that feature was replaced by `elo_diff`, and the prediction
    # came back confident, unchanged, and wrong rather than raising.
    allowed = set(FEATURES) | {"seconds_remaining", "elo_home_win_prob"}
    unknown = sorted(set(kwargs) - allowed)
    if unknown:
        raise ValueError(
            f"unknown feature(s) {unknown}; valid names are {sorted(allowed)}"
        )

    lengths = {len(np.atleast_1d(v)) for v in kwargs.values()}
    n = max(lengths)
    data = {k: np.broadcast_to(np.atleast_1d(v), (n,)).copy() for k, v in kwargs.items()}
    df = pd.DataFrame(data)

    # Broadcasting a pandas nullable column (home_has_ball carries pd.NA where possession is
    # unresolved) produces an object array, which will not cast to float later. Normalise
    # every numeric input here so missing values become plain NaN, which the models accept.
    for col in df.columns:
        if col != "period" or df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col, default in NEUTRAL_DEFAULTS.items():
        if col not in df:
            df[col] = default

    missing_contextual = [c for c in CONTEXTUAL_FIELDS if c not in df.columns]
    if missing_contextual:
        # Recover the clock within the period from total time left, so callers only have to
        # supply seconds_remaining.
        period_arr = df["period"].to_numpy()
        secs = df["seconds_remaining"].to_numpy()
        within = np.where(period_arr <= 4, secs - (4 - np.minimum(period_arr, 4)) * 720, secs)
        ctx = contextual_defaults(period_arr, within)
        for col in missing_contextual:
            df[col] = ctx[col].to_numpy()

    if "home_has_ball" not in df:
        df["home_has_ball"] = np.nan
    if "is_overtime" not in df:
        df["is_overtime"] = df["period"] >= 5

    # Derived from elo_diff so the two can never be passed in disagreeing with each other.
    if "elo_home_win_prob" not in df:
        df["elo_home_win_prob"] = expected_score(df["elo_diff"] + HOME_ADVANTAGE, 0.0)

    # Same formula as features.py. Kept derived rather than passed in.
    df["margin_per_minute_left"] = df["score_margin"] / (df["seconds_remaining"] / 60.0 + 1.0)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"internal error: missing feature columns {missing}")
    return df


# A probability model must never assert certainty. Predictions are clamped so the output can
# read 99.9% but never 100%, and 0.1% but never 0%. Two reasons this matters:
#
#   * It is false. Teams have come back from 20 down with a minute left. Rare is not
#     impossible, and a tool that says 100% is claiming something basketball cannot support.
#   * The raw model already saturates — "up 30 with 0:10 left" returned exactly 1.00000000,
#     and "up 20 with 1:00 left" returned 0.99999479, which rounds to 100% on screen. One is
#     a genuine floating-point saturation, the other a display artefact; the clamp fixes both.
#
# It also protects the scoring: a confident 0 on a game that was actually won is an infinite
# log-loss penalty.
PROBABILITY_FLOOR = 0.001


def win_probability(states: pd.DataFrame, which: str = PRIMARY_MODEL) -> np.ndarray:
    """Home-team win probability for each row of a states frame, never 0 or 1."""
    model = load_model(which)
    X = to_matrix(states)
    p = model.predict_proba(X)[:, 1]
    return np.clip(p, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)


def format_probability(p: float) -> str:
    """Display a probability, showing the clamp as a bound rather than a false precision.

    At the ceiling the honest statement is "at least 99.9%", not "exactly 99.9%" — the model
    declines to be more precise than that, and writing a bare 99.9% claims a precision it does
    not have. Same at the floor.
    """
    if p >= 1.0 - PROBABILITY_FLOOR:
        return f">{(1 - PROBABILITY_FLOOR) * 100:.1f}%"
    if p <= PROBABILITY_FLOOR:
        return f"<{PROBABILITY_FLOOR * 100:.1f}%"
    return f"{p * 100:.1f}%"


def predict_situation(
    score_margin: float,
    period: int,
    clock_seconds: float,
    home_has_ball=None,
    which: str = PRIMARY_MODEL,
    **overrides,
) -> float:
    """Win probability for a single described situation, as the dashboard calculator uses."""
    states = make_states(
        score_margin=float(score_margin),
        seconds_remaining=seconds_remaining_from_clock(period, clock_seconds),
        period=int(period),
        home_has_ball=np.nan if home_has_ball is None else float(bool(home_has_ball)),
        **overrides,
    )
    return float(win_probability(states, which)[0])
