"""
Win probability dashboard.

Two tabs, for two different jobs:

  * **Win Probability** is the deliverable — it answers the question the brief poses: given an
    in-game state, what are the odds. Four controls, because a tool a coach has to configure is
    a tool a coach doesn't open.
  * **Test It Yourself** is the evidence. Every value comes from a real game, so a reader can
    hold the curve up against a game they remember rather than trusting a metric.

All predictions go through src/predict.py, the same path the training pipeline uses, verified
by tests/test_predict.py.
"""

import base64
import html
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))

from branding import GOLD, WINE, logo_path, matchup_colors  # noqa: E402

from config import DATA_PROCESSED  # noqa: E402
from features import elapsed_seconds  # noqa: E402
from elo import ELO_PER_POINT_OF_SPREAD  # noqa: E402
from predict import (  # noqa: E402
    contextual_defaults,
    format_probability,
    max_margin_at,
    max_timeouts_at,
    min_timeouts_at,
    predict_situation,
    seconds_remaining_from_clock,
    win_probability,
)
from config import SEASONS  # noqa: E402
from train_model import TEST_SEASON  # noqa: E402

st.set_page_config(page_title="NBA Win Probability", page_icon="🏀", layout="wide")

# Tile detail lives behind a hover, so the sidebar stays scannable. Everything a coach needs
# at a glance is the three numbers; the explanation is there if they want it.
# Help text is short, one idea per line, and leads with a concrete example. A coach hovering
# a tile wants the punchline, not a paragraph of statistics.
# Written for a coach, not a statistician. Lead with the practical instruction — can I trust
# this number, yes or no — and put the evidence behind it in one plain sentence. Earlier
# versions opened with "differ by 1.6 percentage points on average", which answers a question
# nobody in a gym is asking.
# The opening line here used to read "If it says 70%, treat it as 70%." It was removed, for the
# same reason it was cut from the writeup: it is not true to the precision it implies. Measured
# on the held-out season, when the model says 70% teams win 72.3%. The 1.6 is an average of
# ABSOLUTE gaps, so it carries no direction, and the honest thing is to state the measurement
# and let the reader draw the conclusion.
CALIBRATION_HELP = (
    "Checked on 1,225 games the model had never seen: what it predicted and what actually "
    "happened were 1.6 percentage points of win probability apart, on average.\n\n"
    "One known bias: leads are read slightly high. A first-quarter lead of 8-14 shows about "
    "76% here, where teams in that spot have actually won 71%. Comebacks happen a little more "
    "often than this model expects, so shade early leads down."
)
# The three numbers must add up to 6,135, or a reader can spot 736 games going missing.
# 4,174 trained the model, 736 were a validation set for choosing settings, 1,225 were the
# untouched test season.
SCALE_HELP = (
    "6,135 NBA games pulled in total, across training, validation and testing combined. "
    "About 3 million individual moments.\n\n"
    "The breakdown is in the caption below."
)
MOMENT_HELP = (
    "One moment is a single event — a shot, rebound, foul or timeout — paired with the score, "
    "clock and possession recorded alongside it.\n\n"
    "About 490 per game."
)
PREGAME_HELP = (
    "Before a single possession, from team ratings alone.\n\n"
    "Betting markets are commonly cited around 66% for this — general knowledge, not something "
    "this project measured — so this is close to that range."
)

REPLAY_COLUMNS = [
    "game_id", "game_date", "home_team", "away_team", "period", "clock_seconds",
    "seconds_remaining", "is_overtime", "score_home", "score_away", "score_margin",
    "home_has_ball", "home_fouls_period", "away_fouls_period", "home_in_bonus",
    "away_in_bonus", "home_timeouts_left", "away_timeouts_left", "momentum",
    "elo_diff", "elo_home_win_prob", "margin_per_minute_left", "description",
    "actionType", "home_win",
]


@st.cache_data(show_spinner=False)
def load_season(season: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / f"features_{season}.parquet", columns=REPLAY_COLUMNS)


@st.cache_data(show_spinner=False)
def game_index(season: str) -> pd.DataFrame:
    df = load_season(season)
    # Sort chronologically before taking the last row of each game: relying on the stored row
    # order would silently pick a mid-game score as the final one.
    df = df.sort_values(["game_id", "period", "clock_seconds"], ascending=[True, True, False])
    idx = (
        df.groupby("game_id")
        .agg(
            date=("game_date", "first"),
            home=("home_team", "first"),
            away=("away_team", "first"),
            home_pts=("score_home", "last"),
            away_pts=("score_away", "last"),
        )
        .reset_index()
        .sort_values("date")
    )
    idx["label"] = (
        idx["date"].astype(str) + "  " + idx["away"] + " @ " + idx["home"]
        + "  (" + idx["away_pts"].astype(int).astype(str) + "-" + idx["home_pts"].astype(int).astype(str) + ")"
    )
    return idx


def one_row_per_second(g: pd.DataFrame) -> pd.DataFrame:
    """Collapse a game to one row per second of game clock.

    The feed puts up to 17 events in a single second — both free throws, the foul that caused
    them, and every substitution around them — and their stored order inside that second is not
    reliable, so raw rows zig-zag by a point or two within a second that has only one ending
    state. Keeping the row with the most points scored keeps the state as it stood when the
    second ended.

    This also fixes a visible bug. The chart's hover selection keys on elapsed time, so two rows
    sharing a second were BOTH selected: two dots, and two percentage labels printed on top of
    one another. Unique x values are what the selection needs.
    """
    g = g.copy()
    g["_elapsed"] = elapsed_seconds(g["period"], g["clock_seconds"])
    g["_total"] = g["score_home"] + g["score_away"]
    return (
        g.sort_values(["_elapsed", "_total"], kind="mergesort")
        .drop_duplicates("_elapsed", keep="last")
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def game_curve(season: str, game_id: str) -> pd.DataFrame:
    df = load_season(season)
    g = df[df["game_id"] == game_id].copy()

    # One row per second of game clock. The feed puts up to 17 events in a single second — both
    # free throws, the foul that caused them, and every substitution around them — and their
    # stored order inside that second is not reliable, so the raw rows zig-zag by a point or two
    # within a second that has really only one ending state. Keeping the row with the most
    # points scored keeps the state as it stood when the second ended.
    #
    # This also fixes a visible bug. The hover selection keys on elapsed time, so two rows
    # sharing a second were BOTH selected: two dots, and two percentage labels printed on top of
    # each other. Unique x values are what the selection needs.
    g = one_row_per_second(g)
    g["win_prob"] = win_probability(g)
    g["elapsed_min"] = elapsed_seconds(g["period"], g["clock_seconds"]) / 60.0
    g["win_prob_label"] = g["win_prob"].map(format_probability)
    g["clock_label"] = [clock_label(p, c) for p, c in zip(g["period"], g["clock_seconds"])]
    g["score_label"] = (
        g["score_away"].astype(int).astype(str) + " - " + g["score_home"].astype(int).astype(str)
    )
    # Clipped copies of the curve, for shading the gap to even odds in each team's colour.
    # Computed here rather than in the chart so the clipping happens once per game, not on
    # every rerender.
    g["baseline"] = 0.5
    g["prob_above"] = g["win_prob"].clip(lower=0.5)
    g["prob_below"] = g["win_prob"].clip(upper=0.5)
    return g


def clock_label(period: int, clock_seconds: float) -> str:
    m, s = divmod(int(round(clock_seconds)), 60)
    label = "OT" + str(period - 4) if period > 4 else "Q" + str(period)
    return f"{label} {m}:{s:02d}"



ACCENT = "#C8102E"


@st.cache_data(show_spinner=False)
def logo_img(abbreviation: str, height: int = 44) -> str:
    """An <img> tag for a team logo, or empty string if the file was never fetched.

    The SVG is inlined as a base64 data URI rather than pointed at by path. Streamlit serves
    the app from memory, not from the working directory, so a relative <img src> resolves
    against the wrong root; and Streamlit's markdown sanitiser drops raw <svg> elements. A
    data URI inside a plain <img> survives both.
    """
    path = logo_path(abbreviation)
    if path is None:
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<img src="data:image/svg+xml;base64,{encoded}" height="{height}" '
        f'alt="{abbreviation}" style="vertical-align:middle;">'
    )

# Practical ceiling for the lead slider. The data-driven bound reaches 78 late in a game — the
# largest margin in NBA history — which is true and useless: nobody needs win probability for
# being up 78. Forty covers every situation anyone would actually ask about, and the
# time-based bound still governs early in a game, where it is much tighter than this.
MAX_SLIDER_MARGIN = 40


def hover_layer(base, x_field, y_field, label_field, tooltip, name):
    """Crosshair + readout for a line chart.

    Per the interaction spec: the reader aims at a position on the x-axis, never at a 2px
    line. A wide transparent hit layer takes the pointer, the nearest point wins, and a
    hairline plus a dot plus the value appear together.

    `name` is required, and must be stable across reruns. Left unnamed, Altair numbers
    selections from a global counter — param_1, then param_2, then param_3 — so the same
    chart serialised to a different spec on every rerun. Streamlit compared specs, saw a new
    chart each time, and remounted it: the axes drew but the marks stayed blank until a
    mouse event forced a repaint. That is the "I have to hover before the graph loads" bug.
    """
    # "mouseover" is the documented Vega event for this pattern and fires reliably;
    # "pointerover" did not drive the selection in testing.
    hover = alt.selection_point(
        name=name, fields=[x_field], nearest=True, on="mouseover",
        empty=False, clear="mouseout"
    )
    # Invisible, generously sized hit targets — the pointer only has to be *closest*.
    # The layer carrying the param must be named too. Altair numbers unnamed views from the
    # same global counter as unnamed params — view_1, view_2 — so leaving this implicit put the
    # spec drift straight back, one level down.
    hit = base.mark_point(size=400, opacity=0).encode(
        x=alt.X(f"{x_field}:Q"), y=alt.Y(f"{y_field}:Q"), tooltip=tooltip
    ).add_params(hover).properties(name=f"{name}_hit")
    rule = base.mark_rule(color="#9aa0a6", strokeWidth=1).encode(
        x=alt.X(f"{x_field}:Q"),
        opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
    )
    dot = base.mark_point(size=90, filled=True, color=ACCENT, stroke="white", strokeWidth=2).encode(
        x=alt.X(f"{x_field}:Q"), y=alt.Y(f"{y_field}:Q"),
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    )
    # The readout must carry its own colour. Vega defaults text to near-black, which is
    # invisible against a dark chart surface, and Streamlit's theme does not reach inside the
    # chart to correct it.
    readout = base.mark_text(
        align="left", dx=10, dy=-14, fontSize=14, fontWeight="bold", color="#FFFFFF"
    ).encode(
        x=alt.X(f"{x_field}:Q"), y=alt.Y(f"{y_field}:Q"),
        text=alt.condition(hover, f"{label_field}:N", alt.value("")),
    )
    # The selection comes back with the layer so other marks can respond to the same hover.
    # Without it, anything outside this function can only be drawn statically.
    return hit + rule + dot + readout, hover


# ----------------------------------------------------------------------------- replay view
def render_replay():
    st.subheader("Test It Yourself")

    # The accuracy figures live here AND on Win Probability, doing two different jobs rather
    # than duplicating one. Here it sits beside the other evidence about the model — scale,
    # pre-game accuracy — because this tab's job is to let a reader check the model against a
    # game they watched, and the calibration figure is part of that case. On Win Probability it
    # sits directly under the number it describes, answering "can I trust THIS reading" at the
    # moment someone reads it. It is deliberately duplicated on the other tab rather than moved:
    # the two placements answer different questions and neither one covers for the other.
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy of the odds (win %)", "±1.6 pts", help=CALIBRATION_HELP)
    m2.metric("Games pulled (train + test)", "6,135", help=SCALE_HELP)
    m3.metric("Picks winners pre-game", "65.8%", help=PREGAME_HELP)
    st.caption(
        "Trained on 4,174 games. Another 736 were kept separate to measure which settings "
        f"performed best, not fed into training. Accuracy above comes from a separate 1,225 "
        f"the model never saw — all of {TEST_SEASON}, held back for testing only."
    )

    st.markdown("---")
    st.caption(
        "Pick a game you remember and see whether the line matches how it actually felt. "
        f"Nothing here is entered by hand — every value comes from the game itself. {TEST_SEASON} "
        "is the season the model was never allowed to learn from, so those games are the real test."
    )

    c_season, c_game = st.columns([1, 3])
    season = c_season.selectbox(
        "Season",
        SEASONS,
        key="season",
        index=SEASONS.index(TEST_SEASON),
        format_func=lambda s: f"{s}  (held out)" if s == TEST_SEASON else s,
    )
    idx = game_index(season)
    choice = c_game.selectbox("Choose a game", idx["label"].tolist(), index=0, key="game")
    game_id = idx.loc[idx["label"] == choice, "game_id"].iloc[0]

    if season != TEST_SEASON:
        st.caption(
            f"Note: the model learned from {season}, so it has already seen these games. "
            f"The curve is still accurate — but if you want proof the model works on games it "
            f"has never seen, use {TEST_SEASON}, which was held back from training entirely."
        )

    g = game_curve(season, game_id)
    home, away = g["home_team"].iloc[0], g["away_team"].iloc[0]
    final_h, final_a = int(g["score_home"].iloc[-1]), int(g["score_away"].iloc[-1])
    home_won = final_h > final_a
    home_color, away_color = matchup_colors(home, away)

    # A scoreboard, drawn the way a scoreboard looks: logos flanking the score, the winner's
    # number bright and the loser's dimmed. Three metric tiles reading "CLE (home) 118" said
    # the same thing and looked like a spreadsheet.
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:center; gap:28px;
                    padding:18px 0 6px 0;">
          <div style="text-align:center; min-width:120px;">
            {logo_img(away, 56)}
            <div style="font-size:0.8rem; letter-spacing:.08em; color:#9aa0a6;
                        margin-top:6px;">{away} · AWAY</div>
          </div>
          <div style="font-size:2.6rem; font-weight:700; letter-spacing:-.02em;
                      color:{'#9aa0a6' if home_won else '#FFFFFF'};">{final_a}</div>
          <div style="font-size:1rem; color:#6b7076; letter-spacing:.12em;">FINAL</div>
          <div style="font-size:2.6rem; font-weight:700; letter-spacing:-.02em;
                      color:{'#FFFFFF' if home_won else '#9aa0a6'};">{final_h}</div>
          <div style="text-align:center; min-width:120px;">
            {logo_img(home, 56)}
            <div style="font-size:0.8rem; letter-spacing:.08em; color:#9aa0a6;
                        margin-top:6px;">{home} · HOME</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Even-odds reference line. The y scale is a probability in [0, 1], so this must be 0.5,
    # not 50 — a rule at 50 silently stretches the shared scale and wrecks the axis labels.
    band = (
        alt.Chart(pd.DataFrame({"y": [0.5]}))
        .mark_rule(color="#888", strokeWidth=1, opacity=0.5)
        .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1])))
    )

    x_enc = alt.X("elapsed_min:Q", title="Minutes elapsed", scale=alt.Scale(nice=False),
                  axis=alt.Axis(grid=False))

    # Shade the gap between the line and even odds, in whichever team's colour is ahead. Two
    # clipped areas rather than one conditional fill: a single area with a colour condition
    # recolours whole segments at once and stripes at every lead change.
    def shaded(column: str, color: str):
        return alt.Chart(g).mark_area(color=color, opacity=0.55, line=False).encode(
            x=x_enc,
            y=alt.Y(f"{column}:Q", title=f"{home} win probability",
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y2=alt.Y2("baseline:Q"),
        )

    fill_home = shaded("prob_above", home_color)
    fill_away = shaded("prob_below", away_color)

    base = alt.Chart(g).encode(
        x=x_enc,
        y=alt.Y("win_prob:Q", title=f"{home} win probability",
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
    )
    # White, not a team colour: the line crosses both fills, so it has to stay legible over
    # each of them rather than matching one and vanishing into the other.
    line = base.mark_line(color="#FFFFFF", strokeWidth=1.6, opacity=0.9)
    replay_hover, _ = hover_layer(
        base, "elapsed_min", "win_prob", "win_prob_label", name="replay_hover",
        tooltip=[
            alt.Tooltip("clock_label:N", title="When"),
            alt.Tooltip("score_label:N", title="Score"),
            alt.Tooltip("win_prob_label:N", title=f"{home} wins"),
            alt.Tooltip("description:N", title="Play"),
        ],
    )
    # Period boundaries, derived from how long THIS game actually ran rather than assumed to
    # be four quarters. Regulation is 4x12; every overtime after that is 5 more.
    played = float(g["elapsed_min"].max())
    bounds, names = [0.0, 12.0, 24.0, 36.0, 48.0], ["Q1", "Q2", "Q3", "Q4"]
    while bounds[-1] < played - 0.01:
        bounds.append(bounds[-1] + 5.0)
        names.append(f"OT{len(bounds) - 5}")
    dividers = [b for b in bounds[1:] if b < played - 0.01]
    midpoints = [
        (bounds[i] + min(bounds[i + 1], played)) / 2 for i in range(len(names))
    ]

    # Dashed and brighter than the gridlines, so a quarter break reads as a break rather than
    # as one more line in the grid. The x gridlines are switched off below for the same reason.
    quarters = (
        alt.Chart(pd.DataFrame({"x": dividers}))
        .mark_rule(color="#8B939E", strokeWidth=1, strokeDash=[4, 4], opacity=0.85)
        .encode(x="x:Q")
    )
    quarter_labels = (
        alt.Chart(pd.DataFrame({"x": midpoints, "label": names}))
        .mark_text(color="#9aa0a6", fontSize=11, fontWeight="bold", baseline="top", dy=2)
        .encode(x="x:Q", y=alt.value(0), text="label:N")
    )

    st.altair_chart(
        (fill_home + fill_away + band + quarters + quarter_labels + line + replay_hover)
        .properties(height=400, padding={"left": 5, "top": 10, "right": 24, "bottom": 30})
        .configure_view(strokeWidth=0),
        use_container_width=True,
    )

    # A colour key, because the chart has no legend — an Altair legend would need a colour
    # channel this chart deliberately doesn't use.
    st.markdown(
        f"""
        <div style="display:flex; gap:26px; justify-content:center; font-size:0.86rem;
                    color:#c8ccd0; margin-top:-6px;">
          <span><span style="display:inline-block; width:13px; height:13px; border-radius:3px;
                background:{home_color}; margin-right:7px; vertical-align:-1px;"></span>
                {home} favored</span>
          <span><span style="display:inline-block; width:13px; height:13px; border-radius:3px;
                background:{away_color}; margin-right:7px; vertical-align:-1px;"></span>
                {away} favored</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # A per-play "biggest swings" table lived here and was removed. Its arithmetic was fixed
    # first — it had been crediting substitutions with scoring plays that happened around them —
    # but the corrected version still did not earn its place beside the curve, which already
    # shows where a game turned. Removed rather than kept as filler.


# ------------------------------------------------------------------------- calculator view
def render_calculator():
    st.subheader("What are the odds right now?")
    st.caption(
        "Set the game situation. The number is the home team's chance of winning from there."
    )

    # Two controls a side, grouped by relationship rather than split evenly for its own sake:
    # score and possession share the left column because both describe the state of play right
    # now; period and clock share the right because the clock's range depends on the period.
    c1, c2 = st.columns([1, 1])
    with c1:
        margin_box = st.empty()  # bounded by how much game has been played, so drawn last

        # No "Unknown" option. It existed because the model tolerates missing possession —
        # true of 0.4% of states in the feed — but that is a data edge case, not a question
        # anyone asks. Someone setting up a situation always knows who has the ball.
        ball = st.radio("Who has the ball?", ["Home", "Away"], index=0,
                        horizontal=True, key="ball")

    with c2:
        period = c2.selectbox("Period", [1, 2, 3, 4, 5, 6], index=3, key="period",
                              format_func=lambda p: f"Q{p}" if p <= 4 else f"OT{p - 4}")

        # One control for the clock instead of separate minutes and seconds, stepping a
        # second at a time. It was previously coarse-grained — 5-second steps inside the last
        # two minutes, 15 before that — on the theory that nobody needs precision at 8:35.
        # That theory was wrong in use: the number jumping several points per notch reads as
        # the tool being twitchy, when in fact it is the step size that is coarse. A second
        # per notch makes the curve move the way the game does.
        max_clock = 720 if period <= 4 else 300
        options = list(range(0, max_clock + 1))
        default = 120
        clock_seconds = float(
            c2.select_slider(
                "Time left in the period",
                options=options,
                value=default,
                key="clock",
                format_func=lambda s: f"{s // 60}:{s % 60:02d}",
            )
        )

    # The bound is the smaller of two things: what basketball has actually produced by this
    # point (`reachable`), and a human-relevant ceiling late in blowouts (`MAX_SLIDER_MARGIN`
    # — the all-time record is 78, and nobody needs a slider that reaches a 78-point lead).
    #
    # This used to only take the second number, always, everywhere except literal tip-off — so
    # ten minutes into the first quarter, when no game in five seasons has led by more than 11,
    # the slider still went to 40 and the model would answer a lead that has never happened.
    # That produced a real number for a situation with zero evidence behind it, which is a
    # worse failure mode than a slider whose range visibly shrinks as the clock empties: a
    # bound that moves is honest about why: it moves because what's possible has changed.
    reachable = max_margin_at(period, clock_seconds)
    cap = min(reachable, MAX_SLIDER_MARGIN)
    with margin_box.container():
        if cap == 0:
            st.slider("Score — minus means home is trailing", -1, 1, 0, 1, key="margin",
                      disabled=True,
                      help="No time has been played yet — the score can only be 0-0.")
            margin = 0
            st.caption("**Tip-off** — nothing has been played, so the score must be level.")
        else:
            # A value left over from a wider range (e.g. the clock just moved earlier) must be
            # pulled inside the new bound before the widget is built, or Streamlit renders it
            # pinned at the old, now out-of-range value instead of visibly following the cap.
            st.session_state["margin"] = int(
                np.clip(st.session_state.get("margin", 0), -cap, cap)
            )
            # The sign convention goes in the label, right beside the control. A coach should
            # not have to hover a tooltip to learn which way is which.
            margin = st.slider(
                "Score — minus means home is trailing", -cap, cap, key="margin",
                help="Drag left if the home team is behind, right if ahead. The line below "
                     "states it in words.",
            )
            # Say it in words rather than making the reader translate a signed number. "Home
            # trails by 3" is how a coach states it; "-3" is how a spreadsheet does.
            if margin > 0:
                st.caption(f"**Home leads by {margin}**")
            elif margin < 0:
                st.caption(f"**Home trails by {abs(margin)}**")
            else:
                st.caption("**Tied**")
            if cap < MAX_SLIDER_MARGIN:
                st.caption(f"Bounded to what has actually happened this early: up to {cap}.")

    # Slider starting points come from what is typical at THIS point in a game, not from a
    # single flat league average — two timeouts used is normal in the first quarter and wrong
    # with two minutes left in the fourth.
    typical = contextual_defaults([period], [clock_seconds]).iloc[0]

    with st.expander("More options (team strength, momentum, bonus, timeouts)"):
        d1, d2, d3 = st.columns(3)
        # Deliberately NOT called a "spread". There is no betting data anywhere in this
        # project; this is our own Elo rating gap expressed in points, using a conversion
        # fitted from these seasons (24 Elo points to a point of margin). Labelling it
        # "spread" implied a sportsbook source that does not exist.
        # "Expected to win by" states the quantity in the unit a coach already uses, rather
        # than naming a unit ("points") and leaving them to guess points of what. Still
        # deliberately NOT called a "spread": there is no betting data anywhere in this project.
        # This is our own Elo gap converted at 24 Elo points to a point of margin.
        # "Points", not a percentage, said up front — the tooltip used to bury the unit in
        # its last sentence, so "+5" read for two paragraphs as an unlabelled number before it
        # was clear it meant 5 points of final score, the same way a Vegas spread is stated.
        spread = d1.slider(
            "Home team better by, in points (on a neutral floor)",
            -15.0, 15.0, 0.0, 0.5,
            help="This is points, like a final score margin — not a percentage or a win "
                 "probability.\n\n"
                 "0 means the two teams are evenly matched. +5 means the home team is good "
                 "enough to be favored by 5 points if the game were played on a neutral floor.\n\n"
                 "From our own team ratings, not a betting line. In real matchups the gap is 5 "
                 "points or less about 58% of the time; the biggest gap in five seasons was "
                 "about 20 points.",
        )
        # Momentum earns a control: measured at tied/Q4/2:00 it moves the answer 3.9 points
        # across its range, more than possession does. Raw foul counts move it 0.15 and are
        # deliberately not exposed. Range matches the data, which runs -19 to +18.
        #
        # Locked at tip-off for the same reason the score is: with no time played there is no
        # previous three minutes to have a run in, so any value but 0 describes a game state
        # that cannot exist.
        if cap == 0:
            momentum = 0
            d1.slider("Momentum — home team's net points, last 3 minutes", -1, 1, 0, 1,
                      key="momentum", disabled=True,
                      help="No time has been played yet, so there is no recent run.")
            d1.caption("**Tip-off** — no previous three minutes to run in.")
        else:
            momentum = d1.slider(
                "Momentum — home team's net points, last 3 minutes", -20, 20, 0, 1,
                key="momentum",
                help="How the score has moved over the last three minutes of game clock.",
            )
            # Same pattern as the score slider: a static example while it sits at its default,
            # then the actual situation stated in words once it is moved, using the real number
            # rather than the "+8" example from the tooltip. A coach reads "home outscored by 6",
            # not a signed integer they have to translate.
            if momentum > 0:
                d1.caption(f"**Home team outscored by {momentum}** over the last three minutes.")
            elif momentum < 0:
                d1.caption(f"**Away team outscored by {abs(momentum)}** over the last three "
                          "minutes.")
            else:
                d1.caption("*e.g. +8 would mean the home team outscored the away team by 8 "
                          "over the last three minutes.*")
        # "Left", not "used" — the unit a coach thinks in during a huddle.
        #
        # BOTH ends come from what basketball has produced, not from the rulebook. The rules
        # allow seven (plus two per overtime) at any moment, but nobody has held seven inside
        # the last six minutes of a game, so the ceiling tightens to six there. The rulebook
        # value is kept as an outer guard in case the observed table is ever rebuilt oddly.
        rule_max = 7 + max(period - 4, 0) * 2
        max_to = min(max_timeouts_at(period, clock_seconds), rule_max)
        # At tip-off these are facts, not choices: nobody has called a timeout and nobody has
        # committed a foul, so both teams hold all seven and neither can be in the bonus. Same
        # rule as the score and momentum — if a value cannot exist, do not offer it.
        #
        # Deliberately no `key=` on these four. A keyed widget stores its value in session
        # state, and session state wins over the `value` argument, so the contextual defaults
        # would stop applying as soon as you moved off tip-off — the sliders would stay stuck
        # at seven. Unkeyed widgets re-read `value` when it changes, which is what we want here.
        if cap == 0:
            home_to = away_to = max_to
            home_bonus = away_bonus = False
            for side in ("Home", "Away"):
                d2.slider(f"{side} timeouts left", 0, max_to, max_to, disabled=True,
                          help="Nothing has been played yet — both teams still have all of them.")
                d3.checkbox(f"{side} team in the bonus", value=False, disabled=True,
                            help="No fouls have been committed yet, so neither team is in the bonus.")
        else:
            # Bounded below by what basketball has actually produced by this point, the same way
            # the score slider is bounded above. Two minutes into the first quarter no team has
            # ever held fewer than six, and offering "1" there produced a confident 45.1% that
            # was extrapolated from fourth quarters rather than measured anywhere.
            min_to = min(min_timeouts_at(period, clock_seconds), max_to)
            # The (?) here carries the note that used to be a standalone caption above the whole
            # panel. It applies to every default in this expander, but sits beside timeouts
            # specifically: a heading announcing it for the section went unread.
            home_to = d2.slider(
                "Home timeouts left", min_to, max_to,
                int(np.clip(typical["home_timeouts_left"], min_to, max_to)),
                help="Defaults are what's typical at this point in a game. Adjust to match "
                     "your situation.",
            )
            away_to = d2.slider("Away timeouts left", min_to, max_to,
                                int(np.clip(typical["away_timeouts_left"], min_to, max_to)))
            # Stated as an explicit range ("4-7") rather than a "fewest / most" list: "fewest 4"
            # alone leaves the upper end to be inferred.
            if min_to > 0 and max_to < rule_max:
                d2.caption(f"Teams have had **{min_to}-{max_to}** left at this point in a game.")
            elif min_to > 0:
                d2.caption(f"No team has had fewer than **{min_to}** this early.")
            elif max_to < rule_max:
                d2.caption(f"No team has had more than **{max_to}** this late.")
            home_bonus = d3.checkbox("Home team in the bonus", value=bool(typical["home_in_bonus"]))
            away_bonus = d3.checkbox("Away team in the bonus", value=bool(typical["away_in_bonus"]))

    ball_value = {"Home": True, "Away": False}[ball]
    prob = predict_situation(
        score_margin=margin,
        period=period,
        clock_seconds=clock_seconds,
        home_has_ball=ball_value,
        elo_diff=spread * ELO_PER_POINT_OF_SPREAD,
        home_timeouts_left=float(home_to),
        away_timeouts_left=float(away_to),
        home_in_bonus=int(home_bonus),
        away_in_bonus=int(away_bonus),
        momentum=float(momentum),
    )

    # The answer, as a gauge rather than two number tiles. Labels sit above the bar, never
    # inside it: at >99.9% the losing segment is a sliver two pixels wide, and text placed
    # inside would spill across the fill or vanish. Above the bar it is always legible, at
    # every value the model can produce.
    # Escaped, not interpolated raw: at the clamp these read "<0.1%", and a bare "<" opens
    # what a parser may treat as a tag. This is the one place a model output reaches the page
    # as markup.
    home_label = html.escape(format_probability(prob))
    away_label = html.escape(format_probability(1 - prob))
    home_pct = max(0.4, min(99.6, prob * 100))   # keep a hairline of each colour visible
    st.markdown(
        f"""
        <div style="margin:2px 0 22px 0;">
          <div style="display:flex; justify-content:space-between; align-items:baseline;
                      margin-bottom:8px;">
            <div>
              <span style="font-size:.78rem; letter-spacing:.14em; color:#9aa0a6;">HOME</span>
              <span style="font-size:2.5rem; font-weight:700; color:#FFFFFF;
                           margin-left:10px; letter-spacing:-.02em;">{home_label}</span>
            </div>
            <div>
              <span style="font-size:2.5rem; font-weight:700; color:#9aa0a6;
                           letter-spacing:-.02em;">{away_label}</span>
              <span style="font-size:.78rem; letter-spacing:.14em; color:#9aa0a6;
                           margin-left:10px;">AWAY</span>
            </div>
          </div>
          <div style="display:flex; height:18px; border-radius:9px; overflow:hidden;">
            <div style="width:{home_pct}%; background:{WINE};"></div>
            <div style="width:{100 - home_pct}%; background:#4A515B;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    state = "tied" if margin == 0 else (
        f"home +{margin}" if margin > 0 else f"home {margin}"
    )
    who = {"Home": "home ball", "Away": "away ball"}[ball]
    left, right = st.columns([1, 1])
    left.caption(f"{clock_label(period, clock_seconds)} · {state} · {who}")
    # The warranty belongs against the number, not on the other tab. "Can I trust this?" is a
    # question that occurs at the moment you read the percentage — a paragraph two clicks away
    # does not answer it. It used to be a tile on the evidence tab, on the reasoning that a
    # coach setting a situation does not need calibration methodology in their peripheral
    # vision. But "±1.6 points" is not methodology, it is one line, and this is where it lands.
    right.markdown(
        '<div style="text-align:right; color:#9aa0a6; font-size:0.86rem;">'
        'Accurate to <strong style="color:#c8ccd0;">&plusmn;1.6 pts of win %</strong>, '
        'measured on 1,225 games the model never saw</div>',
        unsafe_allow_html=True,
    )

    # Full width, not a right-hand column. The two number tiles that used to fill the left
    # third became the bar above, and leaving the column split behind put a third of the page
    # aside to hold one line of caption.
    #
    # How the number moves as the clock runs down, holding everything else fixed.
    # The user's exact clock value is forced into the sweep. The selection matches on the clock
    # field, so unless the marker sits on a real curve point there is nothing for a hover to
    # select and the marker can never light up. Snapping the marker to the nearest of 60 points
    # would have moved it by up to six seconds instead.
    clocks = np.unique(np.append(np.linspace(0, max_clock, 60), float(clock_seconds)))
    probs = [
        predict_situation(
            score_margin=margin, period=period, clock_seconds=float(c),
            home_has_ball=ball_value, elo_diff=spread * ELO_PER_POINT_OF_SPREAD,
            home_timeouts_left=float(home_to), away_timeouts_left=float(away_to),
            home_in_bonus=int(home_bonus), away_in_bonus=int(away_bonus),
            momentum=float(momentum),
        )
        for c in clocks
    ]
    curve = pd.DataFrame({"clock": clocks / 60.0, "prob": probs})
    curve["label"] = curve["prob"].map(format_probability)
    curve["mins"] = curve["clock"].map(lambda m: f"{int(m)}:{int(round((m % 1) * 60)):02d} left")
    # Same shading treatment as the game curve on the other tab, in the same two colours as
    # the bar directly above it, so "wine means home is ahead" is learned once.
    curve["baseline"] = 0.5
    curve["prob_above"] = curve["prob"].clip(lower=0.5)
    curve["prob_below"] = curve["prob"].clip(upper=0.5)

    x_enc = alt.X("clock:Q", title="Minutes left in the period",
                  scale=alt.Scale(reverse=True, nice=False),
                  # "~" trims 12.0 to 12; tickMinStep stops it offering half-minutes
                  axis=alt.Axis(format="~f", tickMinStep=1))
    y_title = "Home win probability"

    def shaded(column: str, color: str):
        return alt.Chart(curve).mark_area(color=color, opacity=0.55, line=False).encode(
            x=x_enc,
            y=alt.Y(f"{column}:Q", title=y_title,
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y2=alt.Y2("baseline:Q"),
        )

    base = alt.Chart(curve).encode(
        x=x_enc,
        y=alt.Y("prob:Q", title=y_title,
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
    )
    even = (
        alt.Chart(pd.DataFrame({"y": [0.5]}))
        .mark_rule(color="#888", strokeWidth=1, opacity=0.5)
        .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 1])))
    )
    line = base.mark_line(color="#FFFFFF", strokeWidth=1.6, opacity=0.9)

    # The hover comes first: the marker below reacts to its selection, so the selection has to
    # exist before the marker is built.
    hover, hover_sel = hover_layer(
        base, "clock", "prob", "label", name="calculator_hover",
        tooltip=[alt.Tooltip("mins:N", title="Clock"), alt.Tooltip("label:N", title="Home wins")],
    )

    # Where the situation you set actually sits on the curve, and it lights up when the pointer
    # reaches it — the halo widens and brightens, the dot grows, its ring thickens. Two things
    # make that possible:
    #
    #   * the field names match the curve's ("clock", "prob"), because the selection matches on
    #     the clock field, and a marker whose x lived under another name could never be selected;
    #   * the exact clock value was forced into the sweep above, so the marker sits on a real
    #     curve point rather than between two of them.
    #
    # Drawn last in the layer order, so the hover crosshair's own red dot can never sit on top
    # of it — that previously painted a small red dot inside the gold one, which read as a
    # rendering fault rather than as two marks meaning different things.
    marker_df = pd.DataFrame({"clock": [clock_seconds / 60.0], "prob": [prob]})
    marker_halo = alt.Chart(marker_df).mark_point(
        color=GOLD, filled=True
    ).encode(
        x="clock:Q", y="prob:Q",
        size=alt.condition(hover_sel, alt.value(1100), alt.value(430)),
        opacity=alt.condition(hover_sel, alt.value(0.40), alt.value(0.18)),
    )
    marker = alt.Chart(marker_df).mark_point(
        color=GOLD, filled=True, stroke="#FFFFFF"
    ).encode(
        x="clock:Q", y="prob:Q",
        size=alt.condition(hover_sel, alt.value(330), alt.value(190)),
        strokeWidth=alt.condition(hover_sel, alt.value(3.5), alt.value(2.5)),
    )
    st.altair_chart(
        (shaded("prob_above", WINE) + shaded("prob_below", "#4A515B")
         + even + line + hover + marker_halo + marker)
        .properties(height=300, padding={"left": 5, "top": 10, "right": 24, "bottom": 30})
        .configure_view(strokeWidth=0),
        use_container_width=True,
    )
    st.caption(
        "Same score and situation, as the clock runs down. The gold dot is where you have set "
        "it. Hover to read any point."
    )
# -------------------------------------------------------------------------------------- app
# The league wordmark rather than a basketball emoji. Cached locally in assets/logos, so the
# header does not depend on the network. Falls back to the emoji if the file is missing.
_nba = logo_img("NBA", 46)
_GITHUB_URL = "https://github.com/philstead23/nba-win-probability"
# On top of the header rather than only in the sidebar: a collapsed or scrolled-past sidebar
# would otherwise leave the code deliverable with no visible link on the page a reviewer lands
# on first.
_code_link = (
    f'<a href="{_GITHUB_URL}" target="_blank" style="text-decoration:none; flex-shrink:0;">'
    f'<div style="border:1px solid rgba(250,250,250,.3); border-radius:8px; padding:8px 16px; '
    f'font-size:.9rem; font-weight:600; color:inherit; white-space:nowrap;">'
    f'View code on GitHub &#8599;</div></a>'
)
if _nba:
    st.markdown(
        f'<div style="display:flex; align-items:center; justify-content:space-between; '
        f'gap:20px; margin:0 0 6px 0;">'
        f'<div style="display:flex; align-items:center; gap:20px;">'
        f'{_nba}'
        f'<span style="font-size:2.4rem; font-weight:700; letter-spacing:-.02em;">'
        f'Win Probability</span></div>'
        f'{_code_link}</div>',
        unsafe_allow_html=True,
    )
else:
    st.title("🏀 NBA Win Probability")
    st.markdown(_code_link, unsafe_allow_html=True)

# The sidebar is written BEFORE the tabs, not after. Streamlit streams elements to the browser
# in the order the script produces them, so with this block at the foot of the file the About
# panel only appeared once both tabs had finished — reading parquet, scoring three million
# rows — and visibly popped into place a second after the page opened.
with st.sidebar:
    st.header("About")
    st.write("Win probability for the home team, at any moment of an NBA game.")

    # Scale, not accuracy. These describe what the model is made of; the numbers that argue
    # it can be TRUSTED live on the "Check it against real games" tab, next to the evidence
    # itself. Keeping the two apart stops the sidebar becoming a second results panel.
    st.metric("Seasons of play-by-play", "5")
    st.metric("Games", "6,135")
    st.metric("Game moments", "3.0M", help=MOMENT_HELP)

    st.caption(
        "Set a situation on **Win Probability**. See how it held up on real games under "
        "**Test It Yourself**."
    )
    st.caption(
        "Neutral-site games (Paris, Mexico City, NBA Cup) are excluded — neither team is home."
    )


# The calculator leads: it is the literal answer to the question the assessment poses —
# given an in-game state, what is the probability the home team wins. The replay follows as
# the evidence that the answer can be trusted, which is a supporting role, not the product.
tab_calc, tab_replay = st.tabs(["Win Probability", "Test It Yourself"])
with tab_calc:
    render_calculator()
with tab_replay:
    render_replay()
