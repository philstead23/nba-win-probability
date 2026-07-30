"""
Pulls NBA game lists and play-by-play data via nba_api and caches each season to parquet.

Resumable by design: a season already saved to data/raw is skipped entirely, and within
a season, games already pulled into the season's in-progress cache are skipped too. So a
crash or interruption never forces re-pulling completed work.
"""

import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BACKOFF_BASE_SECONDS,
    DATA_RAW,
    LEAGUE_ID,
    MAX_RETRIES,
    RATE_LIMIT_SECONDS,
    SEASON_TYPE,
    SEASONS,
)
from utils import get_logger

log = get_logger("data_pull")


def call_with_retry(fn, *args, **kwargs):
    """Call an nba_api endpoint, retrying transient failures with exponential backoff."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # nba_api raises plain Exceptions/timeouts on transient failures
            last_exc = exc
            wait = BACKOFF_BASE_SECONDS * (2**attempt)
            log.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed ({exc!r}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Exceeded {MAX_RETRIES} retries") from last_exc


def parse_matchup(matchup: str):
    """'CLE vs. BOS' -> ('CLE', 'BOS'); 'CLE @ BOS' -> ('BOS', 'CLE'). Returns (home, away)."""
    if " vs. " in matchup:
        home, away = matchup.split(" vs. ", 1)
    elif " @ " in matchup:
        away, home = matchup.split(" @ ", 1)
    else:
        raise ValueError(f"Unrecognized MATCHUP format: {matchup!r}")
    return home.strip(), away.strip()


def get_season_game_list(season: str) -> pd.DataFrame:
    """One row per game for the season: game_id, date, home/away teams, final score, winner."""
    raw = call_with_retry(
        leaguegamefinder.LeagueGameFinder,
        season_nullable=season,
        league_id_nullable=LEAGUE_ID,
        season_type_nullable=SEASON_TYPE,
    ).get_data_frames()[0]

    raw["GAME_ID"] = raw["GAME_ID"].astype(str)

    # LeagueGameFinder returns one row per team per game, and the MATCHUP string itself
    # names both teams in home/away order. Parse the string rather than asking "which of
    # the two rows contains 'vs.'": at neutral sites (Mexico City, Paris, NBA Cup
    # semifinals in Las Vegas) the API returns the SAME "@" matchup on BOTH rows, so the
    # row-based test finds no home row at all and silently loses the game.
    per_game = raw.drop_duplicates("GAME_ID")[["GAME_ID", "GAME_DATE", "MATCHUP"]].copy()
    parsed = per_game["MATCHUP"].apply(parse_matchup)
    per_game["home_team"] = [p[0] for p in parsed]
    per_game["away_team"] = [p[1] for p in parsed]

    # Flag games where both team rows carry an identical MATCHUP string, instead of the
    # usual differentiated pair ("CLE vs. BOS" and "BOS @ CLE"). This is the quirk that
    # broke the original row-based home/away test, so it is worth recording.
    #
    # NOTE: this is NOT a neutral-site detector. In 2024-25/2025-26 the affected games all
    # happen to be neutral-site (Mexico City, Paris, NBA Cup semifinals), but neutral-site
    # games in earlier seasons — the 2023-24 Cup semifinals in Las Vegas, the 2024 Paris
    # Game — come back with normal differentiated matchups and are NOT flagged here.
    # Identifying neutral-site games properly would require a venue/arena source.
    variants = raw.groupby("GAME_ID")["MATCHUP"].nunique()
    per_game["ambiguous_matchup"] = per_game["GAME_ID"].isin(set(variants[variants == 1].index))

    team_stats = raw[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS", "WL"]]
    games = per_game.merge(
        team_stats.rename(
            columns={"TEAM_ID": "home_team_id", "PTS": "home_pts", "WL": "home_wl"}
        ),
        left_on=["GAME_ID", "home_team"],
        right_on=["GAME_ID", "TEAM_ABBREVIATION"],
        how="left",
    ).drop(columns="TEAM_ABBREVIATION")
    games = games.merge(
        team_stats.rename(columns={"TEAM_ID": "away_team_id", "PTS": "away_pts"})[
            ["GAME_ID", "away_team_id", "TEAM_ABBREVIATION", "away_pts"]
        ],
        left_on=["GAME_ID", "away_team"],
        right_on=["GAME_ID", "TEAM_ABBREVIATION"],
        how="left",
    ).drop(columns="TEAM_ABBREVIATION")

    games["home_win"] = (games["home_wl"] == "W").astype(int)
    games = games.rename(columns={"GAME_ID": "game_id", "GAME_DATE": "game_date"})
    games["game_id"] = games["game_id"].astype(str)

    # Every game the API returned must survive processing, with both sides resolved.
    expected = raw["GAME_ID"].nunique()
    if len(games) != expected:
        raise RuntimeError(f"{season}: API returned {expected} games but built {len(games)}")
    unresolved = games[games["home_team_id"].isna() | games["away_team_id"].isna()]
    if len(unresolved):
        raise RuntimeError(f"{season}: {len(unresolved)} game(s) missing a home or away side")

    n_ambiguous = int(games["ambiguous_matchup"].sum())
    if n_ambiguous:
        log.info(f"{season}: {n_ambiguous} game(s) with ambiguous MATCHUP strings")

    return games.reset_index(drop=True)


def pull_season_pbp(season: str) -> None:
    season_path = DATA_RAW / f"pbp_{season}.parquet"
    games_path = DATA_RAW / f"games_{season}.parquet"
    partial_path = DATA_RAW / f"pbp_{season}.partial.parquet"

    if season_path.exists():
        log.info(f"{season}: already cached at {season_path.name}, skipping")
        return

    games = get_season_game_list(season)
    games.to_parquet(games_path, index=False)
    log.info(f"{season}: {len(games)} games found")

    already_pulled = set()
    frames = []
    if partial_path.exists():
        prior = pd.read_parquet(partial_path)
        prior["gameId"] = prior["gameId"].astype(str)
        already_pulled = set(prior["gameId"].unique())
        frames.append(prior)
        log.info(f"{season}: resuming, {len(already_pulled)} games already pulled")

    game_ids = games["game_id"].astype(str).tolist()
    remaining = [gid for gid in game_ids if gid not in already_pulled]

    for i, gid in enumerate(remaining, 1):
        df = call_with_retry(playbyplayv3.PlayByPlayV3, game_id=gid).get_data_frames()[0]
        df["gameId"] = df["gameId"].astype(str)
        frames.append(df)

        if i % 25 == 0 or i == len(remaining):
            pd.concat(frames, ignore_index=True).to_parquet(partial_path, index=False)
            log.info(f"{season}: {i}/{len(remaining)} games pulled this run, checkpoint saved")

        time.sleep(RATE_LIMIT_SECONDS)

    full = pd.concat(frames, ignore_index=True)
    full["gameId"] = full["gameId"].astype(str)
    full.to_parquet(season_path, index=False)
    partial_path.unlink(missing_ok=True)
    log.info(f"{season}: done, {full['gameId'].nunique()} games, {len(full)} events -> {season_path.name}")


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        pull_season_pbp(season)


if __name__ == "__main__":
    main()
