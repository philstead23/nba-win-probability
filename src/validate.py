"""
Evaluates the win probability model on the held-out 2025-26 season.

Reports four things:

  1. **Accuracy of the probabilities** — log loss and Brier score, against two reference
     points: a logistic regression, and a constant that always predicts the base rate. A
     model is only meaningfully good if it beats the trivial answer by a clear margin.
  2. **Calibration** — when the model says 70%, do those teams actually win about 70% of
     the time? For a coach this matters more than raw accuracy: a number that cannot be
     taken at face value is not decision-support.
  3. **Real game situations** — model predictions against what actually happened
     historically in specific, genuinely uncertain spots. Deliberately not blowouts: every
     model gets "up 20 with a minute left" right, so those check nothing.
  4. **What the model relies on** — feature importance, to confirm the model is using
     basketball logic and not an artefact.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, MODELS_DIR, REPORTS_DIR
from predict import PROBABILITY_FLOOR
from train_model import FEATURES, LABEL, TEST_SEASON, TRAIN_SEASONS, load, to_matrix
from utils import get_logger

log = get_logger("validate")


def headline_metrics(y, preds: dict) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        rows.append(
            {
                "model": name,
                "log_loss": log_loss(y, p),
                "brier": brier_score_loss(y, p),
                "auc": roc_auc_score(y, p),
            }
        )
    return pd.DataFrame(rows).set_index("model").round(4)


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "pred": p, "actual": np.asarray(y, dtype=float)})
    out = df.groupby("bin").agg(
        n=("pred", "size"), predicted=("pred", "mean"), actual=("actual", "mean")
    )
    out["gap"] = (out["actual"] - out["predicted"]).abs()
    out.index = [f"{edges[i]:.0%}-{edges[i+1]:.0%}" for i in out.index]
    return out.round(4)


def situation_masks(d: pd.DataFrame) -> dict:
    return {
        "Tied, under 30s, home has ball": (
            (d.period == 4) & (d.seconds_remaining <= 30) & (d.score_margin == 0) & (d.home_has_ball == True)
        ),
        "Tied, under 30s, home does NOT have ball": (
            (d.period == 4) & (d.seconds_remaining <= 30) & (d.score_margin == 0) & (d.home_has_ball == False)
        ),
        "Home down 3, under 15s, home ball": (
            (d.period == 4) & (d.seconds_remaining <= 15) & (d.score_margin == -3) & (d.home_has_ball == True)
        ),
        "Home up 5, under 60s, home ball": (
            (d.period == 4) & (d.seconds_remaining <= 60) & (d.score_margin == 5) & (d.home_has_ball == True)
        ),
        "Home up 2, under 60s, away ball": (
            (d.period == 4) & (d.seconds_remaining <= 60) & (d.score_margin == 2) & (d.home_has_ball == False)
        ),
        "Within 3, start of 4th": (
            (d.period == 4) & (d.seconds_remaining.between(700, 720)) & (d.score_margin.abs() <= 3)
        ),
        "Tied entering overtime": ((d.period == 5) & (d.seconds_remaining >= 295)),
        "Home down 10, start of 4th": (
            (d.period == 4) & (d.seconds_remaining.between(690, 720)) & (d.score_margin.between(-11, -9))
        ),
    }


def situation_checks(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Compare model output to the empirical win rate in tight, genuinely uncertain spots."""
    d = df.copy()
    d["pred"] = p
    situations = situation_masks(d)
    rows = []
    for label, mask in situations.items():
        s = d[mask]
        if len(s) < 25:
            rows.append({"situation": label, "games": 0, "model_says": np.nan, "actually_won": np.nan, "ci": ""})
            continue
        # Several states can come from the same game and share its outcome, so the empirical
        # rate and its uncertainty are computed per GAME. Treating each state as independent
        # would overstate the sample several-fold and make noise look like model error.
        per_game = s.groupby("game_id").agg(won=(LABEL, "first"), pred=("pred", "mean"))
        n = len(per_game)
        rate = per_game["won"].mean()
        se = np.sqrt(rate * (1 - rate) / n) if n else np.nan
        rows.append(
            {
                "situation": label,
                "games": n,
                "model_says": round(per_game["pred"].mean() * 100, 1),
                "actually_won": round(rate * 100, 1),
                "ci": f"±{1.96 * se * 100:.0f}",
            }
        )
    out = pd.DataFrame(rows)
    out["difference"] = (out["model_says"] - out["actually_won"]).round(1)
    return out


def season_benchmarks(situations_fn) -> pd.DataFrame:
    """Empirical rate for each situation in EVERY season, not just the test season.

    A single season holds only 50-100 games of any given tight situation, so its rate carries
    an interval of roughly ±10 points. Comparing the model against one season alone invites
    reading normal year-to-year variation as model error — which happened here: the test
    season turned out to be the highest of five on both of the situations that initially
    looked like model failures.
    """
    from config import SEASONS

    rows = {}
    for season in SEASONS:
        df = pd.read_parquet(DATA_PROCESSED / f"features_{season}.parquet")
        df["pred"] = 0.0  # unused; situations_fn only needs the state columns
        for label, mask in situations_fn(df).items():
            s = df[mask]
            if len(s) < 25:
                continue
            per_game = s.groupby("game_id")[LABEL].first()
            rows.setdefault(label, {})[season] = round(per_game.mean() * 100, 1)
    out = pd.DataFrame(rows).T
    out["range"] = out.min(axis=1).astype(str) + " - " + out.max(axis=1).astype(str)
    return out


def main():
    with open(MODELS_DIR / "lgbm.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "logistic.pkl", "rb") as f:
        baseline = pickle.load(f)

    test = load([TEST_SEASON])
    X, y = to_matrix(test), test[LABEL]

    # Clamped exactly as predict.win_probability does, so the reported metrics describe the
    # numbers the dashboard actually shows rather than an unclamped variant of them.
    p_lgb = np.clip(model.predict_proba(X)[:, 1], PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    p_log = np.clip(baseline.predict_proba(X)[:, 1], PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
    base_rate = load(TRAIN_SEASONS)[LABEL].mean()
    p_const = np.full(len(y), base_rate)

    preds = {
        f"always predict {base_rate:.1%}": p_const,
        "logistic regression": p_log,
        "gradient boosted (LightGBM)": p_lgb,
    }

    metrics = headline_metrics(y, preds)
    print("\n" + "=" * 78)
    print(f"HELD-OUT SEASON: {TEST_SEASON}  ({test['game_id'].nunique():,} games, {len(test):,} states)")
    print("=" * 78)
    print(metrics.to_string())
    print("\n(lower log loss and Brier are better; AUC higher is better)")

    for name, p in (("gradient boosted", p_lgb), ("logistic regression", p_log)):
        print("\n" + "=" * 78)
        print(f"CALIBRATION [{name}] — does a stated probability mean what it says?")
        print("=" * 78)
        cal = calibration_table(y, p)
        print(cal.to_string())
        print(f"\nmean absolute calibration gap: {cal['gap'].mean():.4f}")

    print("\n" + "=" * 78)
    print("REAL SITUATIONS — both models vs. what actually happened")
    print("=" * 78)
    sit = situation_checks(test, p_lgb)
    sit_log = situation_checks(test, p_log)
    sit = sit.rename(columns={"model_says": "boosted", "difference": "boosted_diff"})
    sit["logistic"] = sit_log["model_says"]
    sit["logistic_diff"] = sit_log["difference"]
    sit = sit[["situation", "games", "actually_won", "ci", "boosted", "boosted_diff", "logistic", "logistic_diff"]]
    print(sit.to_string(index=False))

    print("\n" + "=" * 78)
    print("SAME SITUATIONS, EVERY SEASON — how much does this vary year to year?")
    print("=" * 78)
    bench = season_benchmarks(situation_masks)
    print(bench.to_string())
    print(
        "\nRead the model's error against this spread, not against one season. Any single\n"
        "season holds only 50-100 games of a given tight situation."
    )

    print("\n" + "=" * 78)
    print("WHAT THE BOOSTED MODEL RELIES ON (by gain)")
    print("=" * 78)
    names = list(model.booster_.feature_name())
    imp = (
        pd.DataFrame(
            {"feature": names, "gain": model.booster_.feature_importance(importance_type="gain")}
        )
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    imp["share"] = (imp["gain"] / imp["gain"].sum() * 100).round(1)
    print(imp[["feature", "share"]].to_string(index=False))
    cal = calibration_table(y, p_lgb)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "validation.json", "w") as f:
        json.dump(
            {
                "test_season": TEST_SEASON,
                "metrics": metrics.reset_index().to_dict("records"),
                "calibration": cal.reset_index().to_dict("records"),
                "situations": sit.to_dict("records"),
                "importance": imp.to_dict("records"),
            },
            f,
            indent=2,
        )
    log.info(f"wrote {REPORTS_DIR / 'validation.json'}")


if __name__ == "__main__":
    main()
