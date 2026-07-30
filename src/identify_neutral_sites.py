"""
Identifies neutral-site games and verifies home/away attribution against the NBA's own
HOME_TEAM_ID field.

Neutral-site detection is deliberately data-driven rather than a hardcoded list of cities.
A team's true home arena is taken to be the arena hosting the plurality of its home games
that season; any home game played somewhere else is neutral-site. This catches
international games, NBA Cup games at neutral venues, and one-off relocations (weather,
arena conflicts) without anyone having to know the schedule in advance, and it adapts
automatically when a team changes buildings between seasons.

Writes data/interim/game_flags_{season}.parquet with `neutral_site` and `venue_known`.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_INTERIM, DATA_RAW, SEASONS
from utils import get_logger

log = get_logger("neutral")

# Most home games a single arena can host and still be considered a neutral site. Real
# neutral-site arrangements top out around 2-3 games per team-season (the Spurs' Austin
# games are the largest in this sample); a renamed home building hosts far more.
MAX_NEUTRAL_GAMES_PER_ARENA = 4

# Basketball judgment call: these venues count as HOME even though
# they are not the team's primary arena. Both host Spurs "home" games in Texas — the
# Alamodome in San Antonio itself, Moody Center in Austin — and draw a substantially Spurs
# crowd, unlike Paris, Mexico City, Berlin, London or the NBA Cup games in Las Vegas.
HOME_DESPITE_ALTERNATE_VENUE = {"Alamodome", "Moody Center"}


def flag_season(season: str) -> pd.DataFrame:
    games = pd.read_parquet(DATA_RAW / f"games_{season}.parquet")
    venues = pd.read_parquet(DATA_RAW / f"venues_{season}.parquet")
    venues["GAME_ID"] = venues["GAME_ID"].astype(str)

    df = games.merge(
        venues[["GAME_ID", "HOME_TEAM_ID", "ARENA_NAME"]],
        left_on="game_id",
        right_on="GAME_ID",
        how="left",
    ).drop(columns="GAME_ID")

    # Verify our MATCHUP-derived home team against the NBA's own designation.
    checkable = df["HOME_TEAM_ID"].notna()
    mismatches = df[checkable & (df["home_team_id"] != df["HOME_TEAM_ID"])]
    if len(mismatches):
        raise RuntimeError(
            f"{season}: {len(mismatches)} game(s) where our home team disagrees with the "
            f"NBA's HOME_TEAM_ID: {mismatches['game_id'].tolist()[:10]}"
        )

    df["venue_known"] = checkable

    # Distinguishing a neutral site from an arena RENAME is the hard part here. Comparing
    # arena names alone fails badly: Staples Center became Crypto.com Arena mid-2021-22, so
    # a name-only test flags every Lakers and Clippers home game before the rename as
    # neutral. Miami, Orlando, Cleveland, Phoenix and San Antonio all renamed too.
    #
    # The distinction is temporal, not nominal. A rename is permanent — the old name never
    # reappears once the new one starts, so the two names occupy separate stretches of the
    # calendar. A neutral-site game is an interruption: the team plays elsewhere for a night
    # and returns to its own building, so it falls *inside* the run of normal home dates.
    #
    # So: take each team's primary arena (most home games), and flag a game as neutral only
    # if it is at some other arena AND falls within the primary arena's date range.
    known = df[checkable].copy()
    known["_date"] = pd.to_datetime(known["game_date"])

    primary = (
        known.groupby(["home_team", "ARENA_NAME"])
        .size()
        .reset_index(name="n")
        .sort_values(["home_team", "n"], ascending=[True, False])
        .drop_duplicates("home_team")
        .set_index("home_team")["ARENA_NAME"]
    )
    spans = (
        known[known.apply(lambda r: primary.get(r["home_team"]) == r["ARENA_NAME"], axis=1)]
        .groupby("home_team")["_date"]
        .agg(["min", "max"])
    )

    # The date-range test alone is not quite enough, because a rename's transition can be
    # messy: Phoenix's 2025-26 games alternate between "PHX Arena" and "Mortgage Matchup
    # Center" for several weeks, so a few old-name games land inside the new name's range.
    # Volume separates the two cases cleanly — a renamed building still hosts most of a
    # team's season (14 games in that example), while a genuine neutral site hosts one or
    # two. Both conditions are required.
    arena_counts = known.groupby(["home_team", "ARENA_NAME"]).size()

    df["expected_arena"] = df["home_team"].map(primary)
    dates = pd.to_datetime(df["game_date"])
    lo = df["home_team"].map(spans["min"])
    hi = df["home_team"].map(spans["max"])
    n_at_arena = pd.Series(
        list(zip(df["home_team"], df["ARENA_NAME"])), index=df.index
    ).map(arena_counts)

    df["neutral_site"] = (
        df["venue_known"]
        & (df["ARENA_NAME"] != df["expected_arena"])
        & dates.between(lo, hi)
        & (n_at_arena <= MAX_NEUTRAL_GAMES_PER_ARENA)
        & ~df["ARENA_NAME"].isin(HOME_DESPITE_ALTERNATE_VENUE)
    )

    # Surface anything sitting near the volume threshold, so a borderline case is visible
    # rather than silently classified either way.
    borderline = known.groupby(["home_team", "ARENA_NAME"]).size()
    borderline = borderline[
        (borderline > MAX_NEUTRAL_GAMES_PER_ARENA) & (borderline <= MAX_NEUTRAL_GAMES_PER_ARENA + 4)
    ]
    for (team, arena), n in borderline.items():
        if arena != primary.get(team):
            log.warning(f"{season}: {team} played {n} games at {arena} — near neutral-site threshold")

    log.info(
        f"{season}: {int(df['neutral_site'].sum())} neutral-site game(s), "
        f"{int((~df['venue_known']).sum())} with unknown venue, "
        f"{len(df) - int(df['neutral_site'].sum())} usable"
    )
    return df


def main():
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    all_neutral = []
    for season in SEASONS:
        df = flag_season(season)
        df.to_parquet(DATA_INTERIM / f"game_flags_{season}.parquet", index=False)
        n = df[df["neutral_site"]].copy()
        n["season"] = season
        all_neutral.append(n)

    combined = pd.concat(all_neutral, ignore_index=True)
    cols = ["season", "game_date", "MATCHUP", "home_team", "away_team", "ARENA_NAME", "expected_arena"]
    print()
    print("=" * 110)
    print("NEUTRAL-SITE GAMES DETECTED (home team not in its usual arena)")
    print("=" * 110)
    print(combined.sort_values(["season", "game_date"])[cols].to_string(index=False))
    print()
    print(f"TOTAL: {len(combined)} neutral-site games across {len(SEASONS)} seasons")


if __name__ == "__main__":
    main()
