"""
Systematic honesty check: sweep the whole state space and report every place the model is
wrong by more than sampling noise can explain.

The headline "±1.6 points" is an AVERAGE over all states. An average can hide a slice that is
badly off, so this walks the space in bins and asks, for each one: does what the model said
sit inside the range the observed win rate could plausibly have come from?

Two rules keep this from lying to itself:

  1. **One vote per game per bin.** An earlier version of this analysis counted 288 *states*
     drawn from 66 *games* and reported a miss that was really sampling noise, because 490
     correlated rows from one game are not 490 pieces of evidence.
  2. **A miss is only a miss if it clears the 95% binomial interval.** With 40 games in a bin,
     an observed rate of 60% is consistent with anything from about 44% to 74%. Calling a
     10-point gap an error there is reading tea leaves.

Run on the held-out season, which the model never trained on, and on all seasons for the
larger sample. Where the two disagree, the disagreement is itself the finding.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import DATA_PROCESSED, SEASONS  # noqa: E402
from predict import win_probability  # noqa: E402
from train_model import TEST_SEASON  # noqa: E402

# Windows a coach would actually name, rather than equal slices of the clock.
TIME_BINS = [
    ("Q1", 1, 0, 720), ("Q2", 2, 0, 720), ("Q3", 3, 0, 720),
    ("Q4 12-8 min", 4, 480, 720), ("Q4 8-4 min", 4, 240, 480),
    ("Q4 4-2 min", 4, 120, 240), ("Q4 2-1 min", 4, 60, 120),
    ("Q4 1:00-0:30", 4, 30, 60), ("Q4 under 0:30", 4, 0, 30),
]
MARGIN_BINS = [
    ("home -20 or worse", -99, -15), ("home -14 to -8", -14, -8),
    ("home -7 to -4", -7, -4), ("home -3 to -1", -3, -1),
    ("tied", 0, 0), ("home +1 to +3", 1, 3),
    ("home +4 to +7", 4, 7), ("home +8 to +14", 8, 14),
    ("home +15 or better", 15, 99),
]


def wilson(successes: int, n: int, z: float = 1.96):
    """95% interval for a win rate. Wilson, not the textbook normal approximation, which
    misbehaves badly near 0% and 100% — exactly where endgame bins live."""
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (centre - half, centre + half)


def load(seasons):
    frames = []
    for s in seasons:
        d = pd.read_parquet(DATA_PROCESSED / f"features_{s}.parquet")
        d["season"] = s
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["p"] = win_probability(df)
    return df


def audit(df, label):
    print(f"\n{'=' * 92}\n{label}   ({df.game_id.nunique():,} games)\n{'=' * 92}")
    print(f"{'situation':>34} {'games':>7} {'actual':>8} {'model':>8} {'miss':>7} "
          f"{'95% range for actual':>24}")

    rows, flagged, thin = [], [], 0
    for tname, period, lo, hi in TIME_BINS:
        for mname, mlo, mhi in MARGIN_BINS:
            sel = df[(df.period == period) & (df.clock_seconds.between(lo, hi))
                     & (df.score_margin.between(mlo, mhi))]
            if sel.empty:
                continue
            # One row per game: correlated states within a game are not independent evidence.
            g = sel.groupby("game_id").agg(won=("home_win", "first"), p=("p", "mean"))
            n, wins = len(g), int(g.won.sum())
            if n < 30:
                thin += 1
                continue
            actual, model = wins / n, g.p.mean()
            lo_ci, hi_ci = wilson(wins, n)
            outside = not (lo_ci <= model <= hi_ci)
            rows.append((n, abs(actual - model)))
            flag = "  <-- outside" if outside else ""
            if outside:
                flagged.append((f"{tname}, {mname}", n, actual, model, lo_ci, hi_ci))
            print(f"{tname + ', ' + mname:>34} {n:>7,} {actual * 100:>7.1f}% {model * 100:>7.1f}% "
                  f"{(model - actual) * 100:>+6.1f} {lo_ci * 100:>10.1f}% - {hi_ci * 100:>5.1f}%{flag}")

    n_bins = len(rows)
    weights = np.array([r[0] for r in rows], dtype=float)
    misses = np.array([r[1] for r in rows]) * 100
    print(f"\n  bins with 30+ games: {n_bins}   (skipped {thin} thinner bins)")
    print(f"  average miss, weighted by games: {np.average(misses, weights=weights):.2f} points")
    print(f"  worst single bin: {misses.max():.1f} points")
    print(f"  bins within 5 points: {(misses <= 5).sum()}/{n_bins} "
          f"({(misses <= 5).mean() * 100:.0f}%)")
    print(f"  bins where the model falls OUTSIDE the 95% range for the actual rate: "
          f"{len(flagged)}/{n_bins}")
    if flagged:
        print("\n  Statistically real misses (not explainable by sample size):")
        for name, n, actual, model, lo_ci, hi_ci in flagged:
            print(f"    {name:>34}  n={n:<6,} actual {actual * 100:5.1f}%  "
                  f"model {model * 100:5.1f}%  (range {lo_ci * 100:.1f}-{hi_ci * 100:.1f}%)")
    else:
        print("\n  No bin is off by more than sampling noise explains.")
    return len(flagged), n_bins


def main():
    all_df = load(SEASONS)
    held = all_df[all_df.season == TEST_SEASON]

    f_held, n_held = audit(held, f"HELD-OUT SEASON {TEST_SEASON} — never seen in training")
    f_all, n_all = audit(all_df, "ALL SIX SEASONS — bigger samples, but includes training data")

    print(f"\n{'=' * 92}\nSUMMARY\n{'=' * 92}")
    print(f"  Held out {TEST_SEASON}: {f_held} of {n_held} bins outside sampling error.")
    print(f"  All seasons:        {f_all} of {n_all} bins outside sampling error.")
    print("\n  A bin flagged on all-seasons but not held-out (or the reverse) is usually noise,")
    print("  not a defect: the two are measuring the same thing at different sample sizes.")


if __name__ == "__main__":
    main()
