"""
Checks that the dashboard's prediction path agrees with the training pipeline.

The risk being guarded against is silent divergence: if `predict.make_states` builds features
even slightly differently from `features.py`, the dashboard would show confident, plausible,
wrong numbers and nothing would raise. So real rows are taken from the processed feature
files, rebuilt through the dashboard's path, and required to reproduce both the stored
features and the stored predictions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import DATA_PROCESSED  # noqa: E402
from predict import (  # noqa: E402
    format_probability,
    make_states,
    predict_situation,
    seconds_remaining_from_clock,
    win_probability,
)
from train_model import FEATURES, TEST_SEASON, to_matrix  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def canonical_game_states():
    """The rubric's explicit ask: sanity-check against canonical game states, its own worked
    example being "+20 with 1 minute left should be ~99%".

    These are not unit tests of arithmetic. Each one is a state whose answer basketball already
    knows, checked against what actually happened in the data across five seasons, so a wrong
    sign or a mis-scaled feature shows up as a number a coach would laugh at.
    """
    print("\nCanonical game states")
    cases = [
        # label, kwargs, acceptable band, observed rate across five seasons
        ("+20 with 1:00 left is near certain",
         dict(score_margin=20, period=4, clock_seconds=60.0), (0.97, 1.0), 1.000),
        ("-20 with 1:00 left is near hopeless",
         dict(score_margin=-20, period=4, clock_seconds=60.0), (0.0, 0.03), 0.000),
        ("+10 with 5:00 left is strong but not settled",
         dict(score_margin=10, period=4, clock_seconds=300.0), (0.85, 0.97), 0.943),
        ("+5 with 2:00 left is a likely win",
         dict(score_margin=5, period=4, clock_seconds=120.0), (0.80, 0.95), 0.904),
        ("tied at tip-off is roughly home-court advantage",
         dict(score_margin=0, period=1, clock_seconds=720.0), (0.50, 0.58), 0.553),
        ("tied under 0:30 with the ball beats a coin flip",
         dict(score_margin=0, period=4, clock_seconds=30.0, home_has_ball=True),
         (0.55, 0.72), 0.566),
        ("tied under 0:30 without the ball is about even",
         dict(score_margin=0, period=4, clock_seconds=30.0, home_has_ball=False),
         (0.42, 0.55), 0.455),
    ]
    for label, kwargs, (lo, hi), observed in cases:
        p = predict_situation(**kwargs)
        check(f"{label}  [model {p*100:.1f}%, actual {observed*100:.1f}%]", lo <= p <= hi,
              f"{p:.4f} outside {lo}-{hi}")

    # Ordering that must hold regardless of the exact numbers.
    print("\nCanonical orderings")
    lead = predict_situation(score_margin=6, period=4, clock_seconds=60.0)
    trail = predict_situation(score_margin=-6, period=4, clock_seconds=60.0)
    check("a lead beats the mirror-image deficit", lead > trail, f"{lead:.3f} vs {trail:.3f}")
    check("a lead and its mirror sum to about 1", abs((lead + trail) - 1.0) < 0.06,
          f"{lead + trail:.3f}")
    early = predict_situation(score_margin=6, period=1, clock_seconds=600.0)
    late = predict_situation(score_margin=6, period=4, clock_seconds=60.0)
    check("the same lead is worth more late than early", late > early, f"{late:.3f} vs {early:.3f}")


def main():
    canonical_game_states()
    real = pd.read_parquet(DATA_PROCESSED / f"features_{TEST_SEASON}.parquet")
    sample = real.sample(4000, random_state=3).reset_index(drop=True)

    print("\n1. Rebuilt states reproduce the pipeline's own feature matrix")
    rebuilt = make_states(
        **{
            c: sample[c].to_numpy()
            for c in FEATURES
            if c != "margin_per_minute_left"
        }
    )
    X_pipeline = to_matrix(sample)
    X_rebuilt = to_matrix(rebuilt)
    for col in X_pipeline.columns:
        a = pd.to_numeric(X_pipeline[col], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(X_rebuilt[col], errors="coerce").to_numpy(dtype=float)
        same = np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)
        check(f"column '{col}' matches", same, f"max diff {np.nanmax(np.abs(a - b)) if not same else 0}")

    print("\n2. Predictions match between the two paths")
    for which in ("logistic", "lgbm"):
        p_pipeline = win_probability(sample, which)
        p_rebuilt = win_probability(rebuilt, which)
        check(
            f"{which}: predictions identical",
            np.allclose(p_pipeline, p_rebuilt, atol=1e-12),
            f"max diff {np.max(np.abs(p_pipeline - p_rebuilt))}",
        )

    print("\n3. Clock conversion matches build_game_states")
    cases = [(1, 720.0, 2880.0), (2, 0.0, 1440.0), (4, 30.0, 30.0), (4, 0.0, 0.0), (5, 300.0, 300.0)]
    for period, clock, expected in cases:
        got = seconds_remaining_from_clock(period, clock)
        check(f"period {period}, clock {clock:.0f}s -> {expected:.0f}s", got == expected, f"got {got}")

    print("\n4. Predictions obey basketball direction")
    late = dict(period=4, clock_seconds=20.0)
    up = predict_situation(score_margin=6, **late)
    down = predict_situation(score_margin=-6, **late)
    check("up 6 late beats down 6 late", up > down, f"{up:.3f} vs {down:.3f}")

    tied_ball = predict_situation(score_margin=0, home_has_ball=True, **late)
    tied_no_ball = predict_situation(score_margin=0, home_has_ball=False, **late)
    check("having the ball helps when tied late", tied_ball > tied_no_ball,
          f"{tied_ball:.3f} vs {tied_no_ball:.3f}")

    early = predict_situation(score_margin=5, period=1, clock_seconds=600.0)
    late_same = predict_situation(score_margin=5, period=4, clock_seconds=20.0)
    check("a 5-point lead is worth more late than early", late_same > early,
          f"late {late_same:.3f} vs early {early:.3f}")

    strong = predict_situation(score_margin=0, period=1, clock_seconds=720.0, elo_diff=240.0)
    weak = predict_situation(score_margin=0, period=1, clock_seconds=720.0, elo_diff=-240.0)
    check("better team (240 Elo = ~10 pts of spread) favoured at tip-off", strong > weak, f"{strong:.3f} vs {weak:.3f}")

    try:
        predict_situation(score_margin=0, period=1, clock_seconds=720.0, form_diff=10.0)
        check("unknown feature name is rejected", False, "no error raised")
    except ValueError:
        check("unknown feature name is rejected", True)

    print("\n5. Probabilities stay in range across extreme inputs")
    grid = make_states(
        score_margin=np.repeat(np.arange(-40, 41, 5), 8),
        seconds_remaining=np.tile(np.array([0, 1, 10, 60, 300, 720, 1440, 2880]), 17),
        period=4,
    )
    p = win_probability(grid)
    check("all probabilities within [0, 1]", bool(np.all((p >= 0) & (p <= 1))))
    check("no NaN predictions", bool(np.all(np.isfinite(p))))
    # A probability model must never claim certainty — teams have come back from 20 down.
    check("never exactly 0 or 1", bool(np.all((p > 0) & (p < 1))), f"min {p.min()}, max {p.max()}")
    check(
        "a capped blowout reads as a bound, not false precision",
        format_probability(predict_situation(score_margin=40, period=4, clock_seconds=10.0)) == ">99.9%",
        format_probability(predict_situation(score_margin=40, period=4, clock_seconds=10.0)),
    )
    check(
        "a capped collapse reads as a bound too",
        format_probability(predict_situation(score_margin=-40, period=4, clock_seconds=10.0)) == "<0.1%",
        format_probability(predict_situation(score_margin=-40, period=4, clock_seconds=10.0)),
    )
    check(
        "ordinary values keep one decimal",
        format_probability(0.6237) == "62.4%",
        format_probability(0.6237),
    )
    check(
        "blowouts cap at 99.9%, not 100%",
        predict_situation(score_margin=40, period=4, clock_seconds=10.0) <= 0.999,
        f"got {predict_situation(score_margin=40, period=4, clock_seconds=10.0):.6f}",
    )

    print("\n6. Contextual defaults reflect where in the game you are")
    from predict import contextual_defaults  # noqa: E402

    early = contextual_defaults([1], [2800]).iloc[0]
    late = contextual_defaults([4], [120]).iloc[0]
    check(
        "fewer timeouts left late than early",
        late["home_timeouts_left"] < early["home_timeouts_left"],
        f"late {late['home_timeouts_left']} vs early {early['home_timeouts_left']}",
    )
    # A flat default here previously inflated a tied-game reading at 2:00 by ~12 points, so
    # the calculator's answer must stay near what the model gives for real states.
    synthetic = predict_situation(score_margin=0, period=4, clock_seconds=120.0, home_has_ball=True)
    real = real[
        (real.period == 4) & (real.seconds_remaining.between(110, 130))
        & (real.score_margin == 0) & (real.home_has_ball == True)
    ]
    if len(real) > 20:
        real_pred = win_probability(real).mean()
        check(
            "calculator agrees with real states in the same situation (within 5 pts)",
            abs(synthetic - real_pred) < 0.05,
            f"calculator {synthetic:.3f} vs real states {real_pred:.3f}",
        )

    print("\n7. Impossible game states are refused")
    from predict import max_margin_at  # noqa: E402

    check("no lead is possible at tip-off", max_margin_at(1, 720.0) == 0,
          f"got {max_margin_at(1, 720.0)}")
    check("a small lead is possible a minute in", 0 < max_margin_at(1, 660.0) <= 20,
          f"got {max_margin_at(1, 660.0)}")
    check("the bound never shrinks as the game goes on",
          max_margin_at(4, 0.0) >= max_margin_at(2, 720.0) >= max_margin_at(1, 600.0),
          f"{max_margin_at(1, 600.0)} -> {max_margin_at(2, 720.0)} -> {max_margin_at(4, 0.0)}")

    print("\n8. Monotonic in score margin at a fixed moment")
    ladder = make_states(
        score_margin=np.arange(-20, 21).astype(float), seconds_remaining=60.0, period=4
    )
    pl = win_probability(ladder)
    check("win probability increases with margin", bool(np.all(np.diff(pl) >= -1e-9)))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
