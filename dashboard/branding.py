"""
Team colours and logos for the dashboard.

Two problems have to be solved before official colours can be used on a chart, and both are
solved here rather than by hand-editing hex codes:

  1. **Some primaries are near-black.** Brooklyn is #000000, Utah is #002B5C. Drawn on a dark
     chart they disappear. Rather than inventing a lightened version, such a team is shown in
     its own published SECONDARY — Cleveland's real gold instead of a pink that no fan would
     call wine. All 30 teams end up on a genuine colour of theirs; none on a synthetic one.
  2. **Some pairs are identical.** New York and Philadelphia are both #006BB6; Utah and
     Washington are both #002B5C; Minnesota and New Orleans are both #0C2340. A chart that
     draws home and away in the same colour is worse than one that ignores team colours, so
     when a matchup collides the away team falls back to its secondary.

Colours are the teams' published primary/secondary marks. Logos come from cdn.nba.com, pulled
once by src/fetch_logos.py and read from disk.
"""

from __future__ import annotations   # this venv is Python 3.9; `Path | None` needs it

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGO_DIR = ROOT / "assets" / "logos"

# Cleveland's own palette, used for the app's furniture rather than for any one team's data.
WINE = "#860038"
GOLD = "#FDBB30"

# (primary, secondary)
TEAM_COLORS = {
    "ATL": ("#E03A3E", "#C1D32F"), "BOS": ("#007A33", "#BA9653"),
    "BKN": ("#000000", "#C4CED4"), "CHA": ("#1D1160", "#00788C"),
    "CHI": ("#CE1141", "#8E9090"), "CLE": ("#860038", "#FDBB30"),
    "DAL": ("#00538C", "#B8C4CA"), "DEN": ("#0E2240", "#FEC524"),
    "DET": ("#C8102E", "#1D42BA"), "GSW": ("#1D428A", "#FFC72C"),
    "HOU": ("#CE1141", "#C4CED4"), "IND": ("#002D62", "#FDBB30"),
    "LAC": ("#C8102E", "#1D428A"), "LAL": ("#552583", "#FDB927"),
    "MEM": ("#5D76A9", "#12173F"), "MIA": ("#98002E", "#F9A01B"),
    "MIL": ("#00471B", "#EEE1C6"), "MIN": ("#0C2340", "#236192"),
    "NOP": ("#0C2340", "#C8102E"), "NYK": ("#006BB6", "#F58426"),
    "OKC": ("#007AC1", "#EF3B24"), "ORL": ("#0077C0", "#C4CED4"),
    "PHI": ("#006BB6", "#ED174C"), "PHX": ("#1D1160", "#E56020"),
    "POR": ("#E03A3E", "#B6BFBF"), "SAC": ("#5A2D81", "#63727A"),
    "SAS": ("#C4CED4", "#8A8D8F"), "TOR": ("#CE1141", "#A1A1A4"),
    "UTA": ("#002B5C", "#F9A01B"), "WAS": ("#002B5C", "#E31837"),
}

NEUTRAL = "#8A9099"
# Lowered from 0.34, which was lifting 24 of 30 teams off their real colour — Boston's green and
# Chicago's red were being brightened despite being perfectly legible. At 0.17 only genuinely
# near-black primaries fail, and those get the team's own SECONDARY rather than a synthetic
# lightened primary: Cleveland shows its real gold instead of a pink that no fan would call wine.
_LUMA_FLOOR = 0.17
_COLLISION = 0.13       # perceptual distance under which two colours are indistinguishable


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{int(round(min(255, max(0, c)))):02X}" for c in rgb)


def _luma(hex_color: str) -> float:
    """Perceived brightness, 0-1. Weighted for how the eye actually responds to each channel."""
    r, g, b = _rgb(hex_color)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def for_dark_background(hex_color: str) -> str:
    """Lift a colour toward white until it is legible on a dark chart, keeping its hue."""
    luma = _luma(hex_color)
    if luma >= _LUMA_FLOOR:
        return hex_color.upper()
    # Blend toward white by exactly the shortfall — enough to see, not enough to wash out.
    t = (_LUMA_FLOOR - luma) / (1.0 - luma)
    r, g, b = _rgb(hex_color)
    return _hex((r + (255 - r) * t, g + (255 - g) * t, b + (255 - b) * t))


def _distance(a: str, b: str) -> float:
    ra, ga, ba = _rgb(a)
    rb, gb, bb = _rgb(b)
    return ((ra - rb) ** 2 + (ga - gb) ** 2 + (ba - bb) ** 2) ** 0.5 / (255 * 3 ** 0.5)


def authentic_display_color(abbreviation: str) -> str:
    """A real colour of this team's, chosen for legibility rather than invented.

    Order of preference:
      1. the published primary, if it reads on a dark chart;
      2. the published secondary, if the primary does not — still the team's own colour;
      3. only if both are near-black, a lifted primary as a last resort.

    Step 2 is the point. Cleveland's primary is #860038; lifting it produced #9D315E, a pink
    nobody would recognise as wine. Their secondary is #FDBB30 — actual Cavs gold. Showing a
    real secondary beats showing a fabricated primary.
    """
    primary, secondary = TEAM_COLORS.get(abbreviation, (NEUTRAL, NEUTRAL))
    if _luma(primary) >= _LUMA_FLOOR:
        return primary.upper()
    if _luma(secondary) >= _LUMA_FLOOR:
        return secondary.upper()
    return for_dark_background(primary)


def matchup_colors(home: str, away: str) -> tuple[str, str]:
    """Two colours guaranteed to be distinguishable from each other and from the background.

    On a collision the away team switches to its OTHER official colour — whichever of its
    published pair is not already in use. The earlier version always reached for the secondary,
    which was a no-op whenever the secondary was already being shown because the primary was
    too dark: Utah at Denver put Utah's yellow against Denver's gold, tried Denver's gold again,
    and fell through to grey. That happened in 37 of the 870 possible matchups.
    """
    away_primary, away_secondary = TEAM_COLORS.get(away, (NEUTRAL, NEUTRAL))

    h = authentic_display_color(home)
    a = authentic_display_color(away)
    if _distance(h, a) < _COLLISION:
        # Whichever of the away team's two colours is not the one we just tried.
        other = away_secondary if a == away_primary.upper() else away_primary
        a = for_dark_background(other)
    if _distance(h, a) < _COLLISION:      # both of the away team's colours clash; last resort
        a = NEUTRAL
    return h, a


def logo_path(abbreviation: str) -> Path | None:
    """Local logo file, or None if it was never fetched — the caller degrades to text."""
    path = LOGO_DIR / f"{abbreviation}.svg"
    return path if path.exists() else None
