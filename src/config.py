"""Project-wide constants: which seasons we pull, where data lives, and API pacing."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# 5 most recent complete "normal" regular seasons.
# 2019-20 (bubble) and 2020-21 (72-game, empty arenas) are excluded on purpose:
# both were played under conditions unlike a normal NBA season.
SEASONS = [
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]

# One extra season of FINAL SCORES ONLY (no play-by-play), used to seed team strength for
# early-season games. Without it, the model spends roughly the first month of every season
# with no idea which teams are good: a team needs 15 prior games before recent form exists,
# so an early November game between the defending champions and a weak team was being called
# a coin flip decided by home court.
SEED_SEASON = "2020-21"

SEASON_TYPE = "Regular Season"
LEAGUE_ID = "00"  # NBA

# Operational settings, per the NBA API reference doc.
RATE_LIMIT_SECONDS = 0.6
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 2  # 2s, 4s, 8s, 16s
