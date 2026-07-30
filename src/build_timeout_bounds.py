"""
The range of timeouts either team has actually had left, by each point of a game.

Written because the calculator was inviting nonsense questions and answering them with a
straight face. Two minutes into the first quarter it let you set the home team to one timeout
left and reported 45.1% — a 12-point penalty — when across five seasons and 71,158 moments at
that point of a game, *no team has ever had fewer than six*. The model is a logistic
regression: it does not look for comparable games, it evaluates a smooth function anywhere you
point it, including at combinations basketball has never produced. The 45.1% was arithmetic
carried in from fourth quarters, not evidence.

This mirrors `margin_bounds` in train_model.py exactly — same 30-second elapsed buckets, same
monotonicity rule — and is kept as a separate script so the bound can be rebuilt without
retraining the model.

The ceiling exists because of the mirror question: if you never call a timeout, do you lose
them? The answer from the data is no — there is no forfeit rule. Minnesota
carried all seven into the fourth at Utah on 2021-12-31, and the feed numbers their first
timeout of the night "Reg.1" at 9:19 of the fourth, so it is unambiguous. But it happened in
exactly 1 game of 6,135, and nobody has been inside the last six minutes holding seven. What
does exist, and is confirmed exactly here, is the limit of four timeouts *used* in the fourth
quarter: 909 teams used four and not one used five.

Run: python src/build_timeout_bounds.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import DATA_PROCESSED, MODELS_DIR, SEASONS  # noqa: E402
from features import elapsed_seconds  # noqa: E402
from train_model import TRAIN_SEASONS  # noqa: E402

OUTPUT = MODELS_DIR / "timeout_bounds.json"
BUCKET_SECONDS = 30


def main():
    # Built from the training seasons only, for the same reason every other fitted artefact is:
    # the held-out season must not inform anything the tool does.
    frames = [
        pd.read_parquet(
            DATA_PROCESSED / f"features_{s}.parquet",
            columns=["period", "clock_seconds", "home_timeouts_left", "away_timeouts_left"],
        )
        for s in TRAIN_SEASONS
    ]
    df = pd.concat(frames, ignore_index=True)
    print(f"{len(df):,} moments from {len(TRAIN_SEASONS)} training seasons "
          f"({', '.join(TRAIN_SEASONS)})")

    bucket = (elapsed_seconds(df["period"], df["clock_seconds"]) // BUCKET_SECONDS).astype(int)
    # Across BOTH teams: the bound describes what basketball has produced for a team, and the
    # slider applies the same limit to either side.
    grouped = df.assign(_b=bucket).groupby("_b")[["home_timeouts_left", "away_timeouts_left"]]
    observed_floor = grouped.min().min(axis=1)
    observed_ceiling = grouped.max().max(axis=1)

    # The floor may only ever fall as the game goes on: a team that could be down to two by
    # minute 20 can still be down to two at minute 21, even if no game happened to sit exactly
    # there. Without this the floor jitters upward on sparse buckets.
    #
    # The ceiling gets NO such rule. Timeouts are replenished — two per overtime — so a
    # monotonically falling ceiling would be wrong the moment a game goes past regulation.
    # Sparse buckets are handled by carrying the last observed value forward instead.
    floor, ceiling = {}, {}
    run_floor = run_ceiling = None
    for b in range(int(observed_floor.index.max()) + 1):
        if b in observed_floor.index:
            lo, hi = int(observed_floor.loc[b]), int(observed_ceiling.loc[b])
            run_floor = lo if run_floor is None else min(run_floor, lo)
            run_ceiling = hi
        if run_floor is not None:
            floor[str(b)] = run_floor
            ceiling[str(b)] = run_ceiling

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump({"floor": floor, "ceiling": ceiling}, f, indent=2)

    print(f"\nwrote {OUTPUT.name}: {len(floor)} buckets\n")
    print(f"{'game time':>12} {'fewest ever held':>18} {'most ever held':>16}")
    for mins in (0, 2, 6, 12, 18, 24, 36, 42, 44, 46, 47, 48):
        b = str(int(mins * 60 // BUCKET_SECONDS))
        if b in floor:
            print(f"{mins:>9} min {floor[b]:>18} {ceiling[b]:>16}")


if __name__ == "__main__":
    main()
