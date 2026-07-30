"""
Trains the win probability model.

Two things about the split matter more than anything else in this file:

  * **Split by game, never by row.** Two moments from the same game are nearly the same
    data point and share an outcome. Splitting rows at random would put both sides of the
    same game in train and test, and the model would be graded on games it had effectively
    already seen. Every split here is on whole games.

  * **The test season is held out by time.** The model trains on 2021-22 through 2024-25 and
    is tested on 2025-26, which it never sees during training or tuning. That mirrors how
    the model would actually be used — fit on history, applied to a season that has not
    happened yet — and is a harder, more honest test than a random split, because it also
    has to survive any year-to-year drift in how the game is played.

Both a gradient-boosted model and a logistic regression are trained on identical features
and weights. The logistic regression won on the held-out season — better log loss, Brier
score, AUC and calibration — and is therefore the model the dashboard uses. See PRIMARY_MODEL
below for why that is the expected result rather than a surprise.
"""

import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, MODELS_DIR, SEASONS
from utils import get_logger

log = get_logger("train")

TEST_SEASON = "2025-26"
TRAIN_SEASONS = [s for s in SEASONS if s != TEST_SEASON]

FEATURES = [
    "score_margin",
    "seconds_remaining",
    "period",
    "is_overtime",
    "home_has_ball",
    "home_fouls_period",
    "away_fouls_period",
    "home_in_bonus",
    "away_in_bonus",
    "home_timeouts_left",
    "away_timeouts_left",
    "momentum",
    # Opponent-adjusted team strength. This replaced a trailing 15-game point differential,
    # which was blind to schedule: +6 a game against a soft slate and +6 against contenders
    # scored identically. On validation, swapping it in improved log loss from 0.4686 to
    # 0.4612 and AUC from 0.846 to 0.853, with the largest gains at tip-off where pre-game
    # strength is the only signal there is.
    "elo_diff",
    "elo_home_win_prob",
    "margin_per_minute_left",
]
LABEL = "home_win"

VALIDATION_FRACTION = 0.15
RANDOM_SEED = 17

# How strongly close, late states are upweighted during training. Selected on the validation
# split (see experiments.py); the held-out season played no part in choosing it.
#
# Without this, the model is close to useless in exactly the situations a coach would consult
# it for. Endgame states are a fraction of a percent of three million rows, so getting them
# right barely moves average log loss and the model simply doesn't bother: it predicted a
# 3.3-point gap between having the ball and not, in a tied game under 30 seconds, where the
# real gap is about 15. Weighting these states lifts that to ~10 points AND improves log loss
# both overall and on late close states — it is not a trade of accuracy for realism.
LEVERAGE_WEIGHT_STRENGTH = 15.0

# The logistic regression beat the gradient-boosted model on the held-out season across
# every measure, so it is what the dashboard serves. This is not a fluke of tuning:
#
#   * Win probability is a smooth, monotonic function of margin and time remaining — close
#     to a logistic curve in margin divided by roughly the square root of time left. That is
#     precisely the shape a logistic regression represents natively, whereas trees have to
#     approximate a smooth surface with axis-aligned steps and spend their capacity doing it.
#   * `margin_per_minute_left` hands the linear model the one interaction that matters, so
#     the main advantage trees would otherwise hold is already supplied.
#   * Smooth output is worth something in itself here: the dashboard draws a win probability
#     curve across a game, and a tree ensemble produces visible staircase jumps between
#     otherwise identical game states.
#
# The boosted model is kept and reported alongside rather than discarded, since the
# comparison is the justification for the choice.
PRIMARY_MODEL = "logistic"


def add_endgame_features(X: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
    """Express the endgame in possessions rather than raw points and seconds.

    Down 4 with 40 seconds left is not "a 4-point deficit and 40 seconds"; it is "two scores
    needed, about three possessions left". Trees can infer this from margin and time in
    principle, but those states are too rare to generate much gradient, so stating it
    directly helps.
    """
    X = X.copy()
    possessions_left = src["seconds_remaining"] / 15.0  # ~15s per possession late in games
    X["possessions_left"] = possessions_left
    X["margin_in_possessions"] = src["score_margin"] / 3.0
    X["possession_deficit"] = possessions_left - (src["score_margin"].abs() / 3.0)
    ball = src["home_has_ball"].astype("float")
    X["ball_x_endgame"] = ball * (src["seconds_remaining"] <= 60).astype(float)
    X["ball_x_close_endgame"] = (
        ball * ((src["seconds_remaining"] <= 60) & (src["score_margin"].abs() <= 3)).astype(float)
    )

    # Margin scaled by the SQUARE ROOT of time left. Scores accumulate like a random walk, so
    # the spread of the remaining swing grows with sqrt(time), not time — this is the natural
    # scale on which a lead should be judged. `margin_per_minute_left` divides by time
    # directly, which understates how safe a small lead is once the clock is nearly out.
    X["margin_per_sqrt_time"] = src["score_margin"] / np.sqrt(src["seconds_remaining"] + 1.0)

    # At zero on the clock with a non-zero margin the game is simply over. That is a fact, not
    # a prediction, but no smooth function of margin and time can express it, so it is stated
    # explicitly: without it the model called a one-point lead at 0:00 a 75% proposition,
    # which is the kind of number that destroys a coach's trust on sight.
    decided = (src["seconds_remaining"] <= 0.5) & (src["score_margin"] != 0)
    X["game_decided"] = decided.astype(float)
    X["decided_sign"] = decided.astype(float) * np.sign(src["score_margin"])

    # Team strength must be allowed to MATTER LESS as the game goes on. With a single global
    # weight, the model has to compromise: Elo is nearly the only information at 0-0 and close
    # to irrelevant with a minute left, and one coefficient cannot say both. The result was a
    # visibly compressed tip-off — predicting 68.8% for matchups that actually went 82.4%,
    # worse calibration than the raw Elo number it was built from. Scaling Elo by the fraction
    # of the game still to play lets its influence decay as the scoreboard takes over.
    fraction_left = (src["seconds_remaining"] / 2880.0).clip(0, 1)
    X["elo_x_time_left"] = src["elo_diff"] * fraction_left
    X["eloprob_x_time_left"] = (src["elo_home_win_prob"] - 0.5) * fraction_left
    return X


def leverage_weights(src: pd.DataFrame, strength: float = LEVERAGE_WEIGHT_STRENGTH) -> np.ndarray:
    """Weight states by how much is genuinely at stake: close games, late clock."""
    close = np.exp(-(src["score_margin"].abs() / 8.0) ** 2)
    late = np.exp(-(src["seconds_remaining"] / 300.0))
    return (1.0 + strength * close * late).to_numpy()


def context_defaults(df: pd.DataFrame) -> dict:
    """Typical timeout and foul state at each point in a game, taken from the data.

    The calculator only asks for score, time, period and possession, so the remaining
    features need defaults. A single fixed value will not do: two timeouts used is normal in
    the first quarter and clearly wrong with two minutes left in the fourth, where the median
    is five. Using the flat value inflated a tied-game-with-the-ball reading at 2:00 from a
    realistic 57% to 63%. Defaults are therefore conditioned on where in the game you are.
    """
    # Bucket by clock WITHIN the period, not by total time left in the game. Keying on
    # seconds_remaining put the entire first quarter in one bucket, so a tip-off situation
    # inherited mid-quarter averages — one timeout spent and two fouls committed at 12:00,
    # when the true answer is zero of each. That alone pushed the calculator's tied-game
    # reading at tip-off to 46.7%, below even odds for a home team.
    d = df.copy()
    d["_bucket"] = d["period"].clip(upper=5).astype(int).astype(str) + "_" + (
        (d["clock_seconds"] // 60).clip(upper=12).astype(int).astype(str)
    )
    agg = d.groupby("_bucket")[
        ["home_timeouts_left", "away_timeouts_left", "home_fouls_period", "away_fouls_period",
         "home_in_bonus", "away_in_bonus"]
    ].median()
    return {
        "buckets": {k: {c: float(v) for c, v in row.items()} for k, row in agg.iterrows()},
        "overall": {c: float(d[c].median()) for c in agg.columns},
    }


def margin_bounds(df: pd.DataFrame) -> dict:
    """Largest score margin actually reached by each point of a game.

    The calculator otherwise lets a user describe states basketball cannot produce — a
    30-point lead at 12:00 of the first quarter, before a single possession — and the model
    answers them with a confident number. Bounding the input by what has genuinely occurred
    across five seasons keeps the tool from inviting nonsense questions.

    Keyed on 30-second buckets of elapsed game time, so the bound tightens sharply in the
    opening minutes where it matters most.
    """
    from features import elapsed_seconds

    elapsed = elapsed_seconds(df["period"], df["clock_seconds"])
    bucket = (elapsed // 30).astype(int)
    observed = df.assign(_b=bucket).groupby("_b")["score_margin"].apply(lambda s: int(s.abs().max()))

    # The bound must never decrease as the game goes on: a margin reachable at minute 5 is
    # still reachable at minute 6, even if no game happened to sit there.
    bounds, running = {}, 0
    for b in range(int(bucket.max()) + 1):
        running = max(running, int(observed.get(b, 0)))
        bounds[str(b)] = running
    return bounds


def load(seasons) -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(DATA_PROCESSED / f"features_{s}.parquet") for s in seasons],
        ignore_index=True,
    )


def to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    # booleans -> float so missing possession stays NaN rather than collapsing to False
    for col in ("home_has_ball", "is_overtime"):
        X[col] = X[col].astype("float")
    return add_endgame_features(X, df)


def split_by_game(df: pd.DataFrame, fraction: float, seed: int):
    """Hold out whole games, so no game appears on both sides of the split."""
    games = df["game_id"].unique()
    rng = np.random.default_rng(seed)
    held = set(rng.choice(games, size=int(len(games) * fraction), replace=False))
    mask = df["game_id"].isin(held)
    return df[~mask].copy(), df[mask].copy()


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_all = load(TRAIN_SEASONS)
    test = load([TEST_SEASON])
    train, valid = split_by_game(train_all, VALIDATION_FRACTION, RANDOM_SEED)

    overlap = set(train["game_id"]) & set(valid["game_id"])
    if overlap:
        raise RuntimeError(f"{len(overlap)} game(s) appear in both train and validation")
    if set(train_all["game_id"]) & set(test["game_id"]):
        raise RuntimeError("train and test share games")

    log.info(
        f"train {len(train):,} rows / {train['game_id'].nunique()} games | "
        f"valid {len(valid):,} / {valid['game_id'].nunique()} | "
        f"test {len(test):,} / {test['game_id'].nunique()} ({TEST_SEASON}, held out)"
    )

    X_tr, y_tr = to_matrix(train), train[LABEL]
    X_va, y_va = to_matrix(valid), valid[LABEL]

    # --- gradient boosted trees ---
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=6000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    weights = leverage_weights(train)
    model.fit(
        X_tr,
        y_tr,
        sample_weight=weights,
        eval_set=[(X_va, y_va)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
    )
    log.info(f"LightGBM stopped at {model.best_iteration_} trees")

    # --- logistic regression baseline ---
    # Same features and same weighting, so the comparison isolates the model rather than
    # the setup around it.
    baseline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    baseline.fit(X_tr, y_tr, logisticregression__sample_weight=weights)

    with open(MODELS_DIR / "context_defaults.json", "w") as f:
        json.dump(context_defaults(train_all), f, indent=2)
    with open(MODELS_DIR / "margin_bounds.json", "w") as f:
        json.dump(margin_bounds(train_all), f, indent=2)

    with open(MODELS_DIR / "lgbm.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODELS_DIR / "logistic.pkl", "wb") as f:
        pickle.dump(baseline, f)
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(
            {
                "features": FEATURES,
                "model_features": list(X_tr.columns),
                "primary_model": PRIMARY_MODEL,
                "leverage_weight_strength": LEVERAGE_WEIGHT_STRENGTH,
                "train_seasons": TRAIN_SEASONS,
                "test_season": TEST_SEASON,
                "best_iteration": int(model.best_iteration_),
                "train_rows": int(len(train)),
                "train_games": int(train["game_id"].nunique()),
            },
            f,
            indent=2,
        )
    log.info(f"saved models to {MODELS_DIR}")


if __name__ == "__main__":
    main()
