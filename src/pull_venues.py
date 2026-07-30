"""
Pulls arena and authoritative home/away designation for every game via ScoreboardV2.

Two purposes:
  1. Identify neutral-site games. LeagueGameFinder's MATCHUP string names a nominal home
     team even for games in Mexico City, Paris or Las Vegas, where no team is actually
     home. Comparing each game's arena against the home team's usual arena finds them.
  2. Independently verify home/away attribution. ScoreboardV2 returns HOME_TEAM_ID
     directly, so it can be checked against the team order parsed out of MATCHUP.

ScoreboardV2 is queried per date rather than per game (~170 dates a season instead of
1,230 games), so this is far cheaper than a per-game endpoint.
"""

import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import scoreboardv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_RAW, RATE_LIMIT_SECONDS, SEASONS
from data_pull import call_with_retry
from utils import get_logger

log = get_logger("venues")

COLUMNS = ["GAME_ID", "GAME_DATE_EST", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "ARENA_NAME"]


def pull_season_venues(season: str) -> pd.DataFrame:
    out_path = DATA_RAW / f"venues_{season}.parquet"
    if out_path.exists():
        log.info(f"{season}: venues already cached, skipping")
        return pd.read_parquet(out_path)

    games = pd.read_parquet(DATA_RAW / f"games_{season}.parquet")
    dates = sorted(games["game_date"].unique())
    log.info(f"{season}: pulling {len(dates)} game dates")

    frames = []
    for i, date in enumerate(dates, 1):
        df = call_with_retry(scoreboardv2.ScoreboardV2, game_date=date).get_data_frames()[0]
        if not df.empty:
            frames.append(df[COLUMNS])
        if i % 50 == 0:
            log.info(f"{season}: {i}/{len(dates)} dates")
        time.sleep(RATE_LIMIT_SECONDS)

    venues = pd.concat(frames, ignore_index=True)
    venues["GAME_ID"] = venues["GAME_ID"].astype(str)
    venues = venues.drop_duplicates("GAME_ID")
    venues.to_parquet(out_path, index=False)
    log.info(f"{season}: {len(venues)} games with venue info -> {out_path.name}")
    return venues


def main():
    for season in SEASONS:
        pull_season_venues(season)


if __name__ == "__main__":
    main()
