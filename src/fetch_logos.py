"""
Downloads the 30 team logos once, into assets/logos/.

The dashboard reads them from disk, never from the network. A submission that fetches images
live looks broken the moment a reviewer opens it behind a firewall or on a plane, and the
whole project's caching discipline exists precisely so nothing has to be re-fetched. Same
contract as the play-by-play pull: skip anything already on disk, rate limit, retry with
backoff.

Source: cdn.nba.com, the same public NBA property the stats API is served from. Logos are
NBA/team marks used here to label their own data.
"""

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nba_api.stats.static import teams as static_teams  # noqa: E402

from config import RATE_LIMIT_SECONDS, MAX_RETRIES, BACKOFF_BASE_SECONDS  # noqa: E402

LOGO_DIR = ROOT / "assets" / "logos"
# "global" is the full-colour primary mark; "primary" serves a stripped one-colour variant.
LOGO_URL = "https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"
# The league wordmark, used once in the dashboard header. White-on-transparent, which suits the
# dark theme; the plain roundel reads as a grey smudge at header size.
LEAGUE_LOGO_URL = "https://cdn.nba.com/logos/nba/nba-logoman-word-white.svg"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch(url: str) -> bytes:
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = BACKOFF_BASE_SECONDS ** (attempt + 1)
            print(f"    retry {attempt + 1}/{MAX_RETRIES - 1} in {wait}s ({exc})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def main():
    LOGO_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = skipped = 0

    for team in sorted(static_teams.get_teams(), key=lambda t: t["abbreviation"]):
        abbr = team["abbreviation"]
        path = LOGO_DIR / f"{abbr}.svg"
        if path.exists() and path.stat().st_size > 0:
            skipped += 1
            continue
        body = fetch(LOGO_URL.format(team_id=team["id"]))
        path.write_bytes(body)
        print(f"  {abbr}  {len(body):>6} bytes")
        downloaded += 1
        time.sleep(RATE_LIMIT_SECONDS)

    league = LOGO_DIR / "NBA.svg"
    if not (league.exists() and league.stat().st_size > 0):
        league.write_bytes(fetch(LEAGUE_LOGO_URL))
        print(f"  NBA  {league.stat().st_size:>6} bytes (league wordmark)")
        downloaded += 1
    else:
        skipped += 1

    print(f"\n{downloaded} downloaded, {skipped} already cached -> {LOGO_DIR}")


if __name__ == "__main__":
    main()
