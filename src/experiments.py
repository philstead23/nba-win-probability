"""
Model-selection experiments, scored on the validation split only.

The held-out 2025-26 season is deliberately untouched here. Choosing a configuration by its
test score would make that score meaningless — it would measure how well the configuration
was picked for that season, not how the model performs on a season it has never seen.

Alongside overall log loss, each variant is scored on late-game states specifically. Those
are under 2% of rows, so a model can post a good average while being useless in exactly the
situations a coach would consult it for.
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_model import (
    FEATURES,
    LABEL,
    RANDOM_SEED,
    TRAIN_SEASONS,
    VALIDATION_FRACTION,
    load,
    split_by_game,
    to_matrix,
)
from utils import get_logger

log = get_logger("experiments")


def evaluate(model, X_va, va, label):
    p = model.predict_proba(X_va)[:, 1]
    y = va[LABEL]
    overall = log_loss(y, p)

    late = (va["period"] >= 4) & (va["seconds_remaining"] <= 120) & (va["score_margin"].abs() <= 5)
    late_ll = log_loss(y[late], p[late]) if late.sum() > 100 else np.nan

    # Does the model separate "has the ball" from "doesn't" in a tied endgame?
    tied = (va["period"] == 4) & (va["seconds_remaining"] <= 30) & (va["score_margin"] == 0)
    with_ball = tied & (va["home_has_ball"] == True)
    without = tied & (va["home_has_ball"] == False)
    spread = (
        p[with_ball].mean() - p[without].mean()
        if with_ball.sum() > 25 and without.sum() > 25
        else np.nan
    )
    actual_spread = (
        y[with_ball].mean() - y[without].mean()
        if with_ball.sum() > 25 and without.sum() > 25
        else np.nan
    )
    return {
        "variant": label,
        "trees": getattr(model, "best_iteration_", None),
        "log_loss": round(overall, 5),
        "late_close_log_loss": round(late_ll, 5),
        "possession_spread": round(spread * 100, 1),
        "actual_spread": round(actual_spread * 100, 1),
    }


def add_endgame_features(X: pd.DataFrame, src: pd.DataFrame) -> pd.DataFrame:
    """Express the endgame the way a coach does: in possessions, not raw points and seconds.

    Down 4 with 40 seconds is not 'a 4-point deficit and 40 seconds'; it is 'two scores with
    about three possessions left'. Trees can in principle infer this from margin and time,
    but the states where it matters are a fraction of a percent of the data, so there is
    almost no gradient pushing them to.
    """
    X = X.copy()
    possessions_left = src["seconds_remaining"] / 15.0  # ~15s per possession late in games
    X["possessions_left"] = possessions_left
    X["margin_in_possessions"] = src["score_margin"] / 3.0
    # Net of the deficit, how many scoring chances does the trailing side actually have?
    X["possession_deficit"] = possessions_left - (src["score_margin"].abs() / 3.0)
    ball = src["home_has_ball"].astype("float")
    X["ball_x_endgame"] = ball * (src["seconds_remaining"] <= 60).astype(float)
    X["ball_x_close_endgame"] = (
        ball * ((src["seconds_remaining"] <= 60) & (src["score_margin"].abs() <= 3)).astype(float)
    )
    return X


def leverage_weights(src: pd.DataFrame, strength: float) -> np.ndarray:
    """Upweight close, late states so they are not drowned out by routine mid-game rows."""
    close = np.exp(-(src["score_margin"].abs() / 8.0) ** 2)
    late = np.exp(-(src["seconds_remaining"] / 300.0))
    return (1.0 + strength * close * late).to_numpy()


def fit(X_tr, y_tr, X_va, y_va, weights=None, **params):
    defaults = dict(
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
    defaults.update(params)
    m = lgb.LGBMClassifier(**defaults)
    m.fit(
        X_tr,
        y_tr,
        sample_weight=weights,
        eval_set=[(X_va, y_va)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
    )
    return m


def main():
    all_train = load(TRAIN_SEASONS)
    tr, va = split_by_game(all_train, VALIDATION_FRACTION, RANDOM_SEED)
    y_tr, y_va = tr[LABEL], va[LABEL]

    base_tr, base_va = to_matrix(tr), to_matrix(va)
    end_tr = add_endgame_features(base_tr, tr)
    end_va = add_endgame_features(base_va, va)

    deeper = dict(num_leaves=127, min_child_samples=150)
    variants = [
        ("A: baseline (no weights)", base_tr, base_va, None, {}),
        ("F: weights x10", base_tr, base_va, leverage_weights(tr, 10), {}),
        ("J: weights x10, deeper", base_tr, base_va, leverage_weights(tr, 10), deeper),
        ("K: weights x20, deeper", base_tr, base_va, leverage_weights(tr, 20), deeper),
        ("L: endgame + weights x20, deeper", end_tr, end_va, leverage_weights(tr, 20), deeper),
        ("M: endgame + weights x15", end_tr, end_va, leverage_weights(tr, 15), {}),
    ]

    rows = []
    for label, X_tr, X_va, w, params in variants:
        m = fit(X_tr, y_tr, X_va, y_va, weights=w, **params)
        rows.append(evaluate(m, X_va, va, label))
        log.info(f"{label}: {rows[-1]}")

    out = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("VALIDATION RESULTS (held-out test season NOT used)")
    print("=" * 100)
    print(out.to_string(index=False))
    print("\npossession_spread = predicted win% with ball minus without, tied game under 30s")
    print("actual_spread     = what actually happened in those same states")


if __name__ == "__main__":
    main()
