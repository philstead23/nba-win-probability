"""
One-off repair: rebuild every season's game list with the fixed home/away parser, then
pull play-by-play only for games the original (buggy) list had dropped.

The original list-builder decided home vs. away by asking which of a game's two rows
contained "vs.". Neutral-site games return the same "@" matchup on both rows, so those
games resolved to no home row and were dropped by the join. This restores them without
re-pulling any play-by-play that already succeeded.
"""

import sys
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playbyplayv3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_RAW, RATE_LIMIT_SECONDS, SEASONS
from data_pull import call_with_retry, get_season_game_list
from utils import get_logger

log = get_logger("repair")


def main():
    for season in SEASONS:
        games = get_season_game_list(season)
        games.to_parquet(DATA_RAW / f"games_{season}.parquet", index=False)

        pbp_path = DATA_RAW / f"pbp_{season}.parquet"
        pbp = pd.read_parquet(pbp_path)
        pbp["gameId"] = pbp["gameId"].astype(str)

        have = set(pbp["gameId"].unique())
        missing = [gid for gid in games["game_id"].astype(str) if gid not in have]
        log.info(f"{season}: {len(games)} games in list, {len(have)} in pbp, {len(missing)} to pull")

        if not missing:
            continue

        frames = [pbp]
        for gid in missing:
            df = call_with_retry(playbyplayv3.PlayByPlayV3, game_id=gid).get_data_frames()[0]
            df["gameId"] = df["gameId"].astype(str)
            if df.empty:
                log.warning(f"{season}: {gid} returned no play-by-play rows")
            frames.append(df)
            time.sleep(RATE_LIMIT_SECONDS)

        full = pd.concat(frames, ignore_index=True)
        full["gameId"] = full["gameId"].astype(str)
        full.to_parquet(pbp_path, index=False)
        log.info(f"{season}: repaired -> {full['gameId'].nunique()} games, {len(full)} events")


if __name__ == "__main__":
    main()
