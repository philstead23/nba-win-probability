"""
Verifies the foul and timeout features against the NBA's own bookkeeping, every season.

These two features are the ones most at risk of being quietly wrong, because neither exists
as a field in the API — both are reconstructed from event rows and from text embedded in
descriptions. Three separate errors were found in them during development:

  * offensive fouls, charges and defensive three-seconds were counted toward the team total
    when the league does not count them (15% of fouls numbered wrongly),
  * the bonus was derived from a fixed threshold of 5, which misses the last-two-minutes rule
    (the league had 793 fouls in the penalty that a count-of-5 rule did not),
  * an attempt to read the league's team-foul number directly would have disabled bonus
    detection entirely, because that number is never published past 4.

The NBA publishes enough in the play-by-play to check all of it: a team-foul number on foul
descriptions up to 4, and a "PN" marker in place of that number once a team is in the penalty.
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import DATA_PROCESSED, DATA_RAW, SEASONS  # noqa: E402
from features import (  # noqa: E402
    NICKNAME_TO_ABBR,
    NON_TEAM_FOULS,
    TIMEOUTS_PER_GAME,
    TIMEOUTS_PER_OVERTIME,
)
from utils import parse_clock  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def chronological(season: str) -> pd.DataFrame:
    p = pd.read_parquet(DATA_RAW / f"pbp_{season}.parquet")
    p = p[p["actionType"] != "Instant Replay"].copy()
    p["secs"] = p["clock"].map(parse_clock)
    return p.sort_values(
        ["gameId", "period", "secs", "actionNumber"], ascending=[True, True, False, True]
    ).reset_index(drop=True)


def main():
    for season in SEASONS:
        print(f"\n=== {season} ===")
        p = chronological(season)

        # ---------------- fouls ----------------
        fouls = p[(p["actionType"] == "Foul") & p["teamId"].ne(0)].copy()
        counted = fouls[~fouls["subType"].isin(NON_TEAM_FOULS)].copy()
        counted["mine"] = counted.groupby(["gameId", "period", "teamId"]).cumcount() + 1
        counted["published"] = counted["description"].str.extract(r"\.T(\d+)\)")[0].astype(float)

        cmp_ = counted.dropna(subset=["published"])
        agree = (cmp_["mine"] == cmp_["published"]).mean() * 100
        # A tiny number of fouls carry a team-foul number inconsistent with the surrounding
        # sequence in the NBA's own feed. The bar is set just below perfect to tolerate those
        # without hiding a genuine regression.
        check(
            "team-foul count matches the NBA's published number",
            agree >= 99.98,
            f"{agree:.2f}% of {len(cmp_):,} numbered fouls",
        )

        fouls["pn"] = fouls["description"].str.contains(r"\.PN\)", regex=True, na=False)
        first_pn = fouls[fouls["pn"]].groupby(["gameId", "period", "teamId"])["secs"].max()
        # every foul after a team's first PN in a period should also be PN
        fouls = fouls.join(first_pn.rename("pn_from"), on=["gameId", "period", "teamId"])
        after = fouls[fouls["pn_from"].notna() & (fouls["secs"] <= fouls["pn_from"])]
        # a foul at or after the penalty starts must be marked PN or be an excluded type
        stray = after[~after["pn"] & ~after["subType"].isin(NON_TEAM_FOULS)]
        rate = len(stray) / max(len(fouls), 1) * 100
        # ~0.04% of fouls are numbered by the NBA after that team was already marked in the
        # penalty — an inconsistency in the league's own bookkeeping, not in this code. The
        # threshold catches a real regression while tolerating that noise floor.
        check(
            "penalty state is consistent within a period",
            rate < 0.1,
            f"{len(stray)} inconsistent of {len(fouls):,} ({rate:.3f}%)",
        )

        # ---------------- timeouts ----------------
        to = p[p["actionType"] == "Timeout"].copy()
        to["abbr"] = (
            to["description"].str.split(" Timeout").str[0].str.strip().str.lower()
            .map(NICKNAME_TO_ABBR)
        )
        check(
            "every timeout row resolves to a team",
            to["abbr"].notna().all(),
            f"{int(to['abbr'].isna().sum())} unresolved of {len(to):,}",
        )

        games = pd.read_parquet(DATA_RAW / f"games_{season}.parquet")
        pairs = games.set_index("game_id")[["home_team", "away_team"]]
        j = to.join(pairs, on="gameId")
        wrong_team = j[(j["abbr"] != j["home_team"]) & (j["abbr"] != j["away_team"])]
        check(
            "no timeout attributed to a team not in that game",
            len(wrong_team) == 0,
            f"{len(wrong_team)} misattributed",
        )

        # The published counter should equal the nth CHARGED timeout for that team. A coach's
        # challenge produces a timeout row but does not increment the counter, so it must be
        # excluded before counting — an earlier version of this check did not, and reported a
        # 9% error rate that was entirely its own.
        to["published"] = to["description"].str.extract(r"(?:Full|Reg\.)\s*(\d+)")[0].astype(float)
        charged = to[to["subType"] != "Coach Challenge"].copy()
        charged["mine"] = charged.groupby(["gameId", "abbr"]).cumcount() + 1
        c = charged.dropna(subset=["published"])
        match = (c["mine"] == c["published"]).mean() * 100
        check(
            "timeout counter matches an independent count of charged timeouts",
            match > 99.9,
            f"{match:.2f}% of {len(c):,} charged timeouts",
        )

        # ---------------- the built features ----------------
        feats = pd.read_parquet(
            DATA_PROCESSED / f"features_{season}.parquet",
            columns=[
                "game_id", "period", "home_timeouts_left", "away_timeouts_left",
                "home_fouls_period", "away_fouls_period", "home_in_bonus", "away_in_bonus",
            ],
        )
        allowance = TIMEOUTS_PER_GAME + (feats["period"] - 4).clip(lower=0) * TIMEOUTS_PER_OVERTIME
        in_range = (
            feats["home_timeouts_left"].between(0, allowance).all()
            and feats["away_timeouts_left"].between(0, allowance).all()
        )
        check("timeouts left stays within the allowance", bool(in_range))

        no_neg = (feats["home_fouls_period"] >= 0).all() and (feats["away_fouls_period"] >= 0).all()
        check("foul counts are never negative", bool(no_neg))

        binary = set(feats["home_in_bonus"].unique()) <= {0, 1}
        check("bonus flag is strictly 0/1", bool(binary))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {sorted(set(FAILURES))}")
        sys.exit(1)
    print("All foul and timeout checks passed, every season.")


if __name__ == "__main__":
    main()
