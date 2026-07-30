"""
Runs the dashboard headlessly and fails on any uncaught exception.

This exists because of a bug that shipped: when timeout defaults moved from a flat dictionary
into context-aware lookups, the calculator kept referencing the removed key and raised
`KeyError: 'home_timeouts_used'` the moment anyone opened that tab. Every other test passed,
because they exercised the prediction path and never imported the dashboard at all.

Streamlit's AppTest executes the real script — widgets, callbacks and all — so a broken
reference surfaces here instead of on screen.

Widgets are addressed by `key`, not by position. Adding the season picker shifted every
selectbox index and broke this file, which is the same brittleness in test form.
"""

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / "dashboard" / "app.py")
TIMEOUT = 180

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def exceptions(at):
    return [f"{e.value}" for e in at.exception]


def main():
    print("\n1. App loads without error")
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    check("no exception on initial load", not at.exception, exceptions(at))

    print("\n2. Both tabs render")
    check("two tabs present", len(at.tabs) == 2, f"found {len(at.tabs)}")
    check("a chart rendered on the replay tab", len(at.get("arrow_vega_lite_chart")) >= 1)
    check("accuracy tiles rendered", len(at.metric) >= 3, f"found {len(at.metric)}")
    check("team logos rendered on the replay tab",
          sum("data:image/svg+xml;base64" in m.value for m in at.markdown) >= 1)

    print("\n3. Calculator controls exist and are wired up")
    # This is the exact surface the KeyError lived on: the calculator's expander contents.
    check("sliders rendered", len(at.slider) >= 3, f"found {len(at.slider)}")
    check("clock select-slider rendered", len(at.select_slider) >= 1)
    check("possession radio rendered", len(at.radio) >= 1)
    check("bonus checkboxes rendered", len(at.checkbox) >= 2, f"found {len(at.checkbox)}")

    print("\n4. Changing inputs re-runs cleanly and moves the number the right way")
    def home_win_pct(app):
        """Read the calculator's headline number out of the rendered bar.

        It used to be an `st.metric` looked up by label. The metric was replaced by a split
        bar drawn in HTML, so this now parses the value the user actually sees. Still keyed
        on content, never on position — index-based lookup has broken this file twice, once
        when a season picker was added and once when sidebar tiles were.
        """
        pattern = re.compile(r"HOME</span>\s*<span[^>]*>\s*(.+?)\s*</span>", re.S)
        for block in app.markdown:
            found = pattern.search(block.value)
            if found:
                # Values at the clamp read as bounds (">99.9%" / "<0.1%"), so strip the
                # comparator before parsing rather than assuming a bare number.
                return float(html.unescape(found.group(1)).lstrip("<>").rstrip("%"))
        return float("nan")

    at.slider(key="margin").set_value(-20).run()
    check("no exception after setting a 20-point deficit", not at.exception, exceptions(at))
    losing = home_win_pct(at)

    at.slider(key="margin").set_value(20).run()
    check("no exception after setting a 20-point lead", not at.exception, exceptions(at))
    winning = home_win_pct(at)

    check("a 20-point lead beats a 20-point deficit", winning > losing, f"{winning} vs {losing}")

    print("\n4c. Momentum caption describes the actual value, not a fixed example")
    def momentum_caption(app):
        for c in app.caption:
            if "outscored" in c.value.lower() or "e.g." in c.value.lower():
                return c.value
        return None
    check("shows a general example at 0",
          "e.g." in (momentum_caption(at) or ""), momentum_caption(at))
    at.slider(key="momentum").set_value(6).run()
    check("describes home outscoring by the real value",
          momentum_caption(at) == "**Home team outscored by 6** over the last three minutes.",
          momentum_caption(at))
    at.slider(key="momentum").set_value(-4).run()
    check("describes away outscoring by the real value",
          momentum_caption(at) == "**Away team outscored by 4** over the last three minutes.",
          momentum_caption(at))
    at.slider(key="momentum").set_value(0).run()

    print("\n5. Possession toggle works without error")
    at.radio(key="ball").set_value("Away").run()
    check("no exception switching possession to away", not at.exception, exceptions(at))
    # "Unknown" was removed deliberately; guard against it drifting back in.
    check("possession offers exactly Home and Away",
          at.radio(key="ball").options == ["Home", "Away"],
          f"found {at.radio(key='ball').options}")
    at.radio(key="ball").set_value("Home").run()
    check("no exception switching possession back to home", not at.exception, exceptions(at))

    print("\n6. Season picker works for training seasons too")
    at.selectbox(key="season").set_value("2022-23").run()
    check("switching to a training season renders", not at.exception, exceptions(at))
    at.selectbox(key="season").set_value("2025-26").run()
    check("switching back to the held-out season renders", not at.exception, exceptions(at))

    print("\n7. Inputs that cannot exist at tip-off are locked")
    # Nothing has been played at Q1 12:00, so the score must be 0-0 and there is no previous
    # three minutes to have had a run in. Both controls are disabled there and nowhere else.
    at.selectbox(key="period").set_value(1).run()
    at.select_slider(key="clock").set_value(720).run()
    check("score locked at tip-off", at.slider(key="margin").disabled)
    check("momentum locked at tip-off", at.slider(key="momentum").disabled)

    at.select_slider(key="clock").set_value(600).run()
    check("score unlocked once time has been played", not at.slider(key="margin").disabled)
    check("momentum unlocked once time has been played", not at.slider(key="momentum").disabled)

    # Start of overtime: the previous three minutes were the end of the 4th, so momentum is
    # meaningful and must NOT be locked.
    at.selectbox(key="period").set_value(5).run()
    at.select_slider(key="clock").set_value(300).run()
    check("momentum available at the start of overtime",
          not at.slider(key="momentum").disabled)

    print("\n7b. Score slider cannot describe leads basketball has never produced")
    # The bound used to only ever be the human ceiling (40), everywhere except literal tip-off,
    # so ten minutes into Q1 — where no game in five seasons led by more than 11 — the slider
    # still reached 40 and the model answered a lead with zero evidence behind it.
    at.selectbox(key="period").set_value(1).run()
    at.select_slider(key="clock").set_value(653).run()   # 10:53 left in Q1
    ms = at.slider(key="margin")
    check("Q1 10:53 caps well below the human ceiling of 40",
          ms.max < 40, f"max was {ms.max}")
    at.slider(key="margin").set_value(40).run()
    check("requesting 40 clamps silently instead of erroring",
          not at.exception, exceptions(at))
    check("clamped value sits at or below the real cap",
          at.slider(key="margin").value <= ms.max, f"{at.slider(key='margin').value}")

    at.selectbox(key="period").set_value(4).run()
    at.select_slider(key="clock").set_value(0).run()
    check("deep in Q4 the human ceiling of 40 still applies (not the record of 78)",
          at.slider(key="margin").max == 40, f"max was {at.slider(key='margin').max}")

    print("\n8. Timeout sliders cannot describe states basketball has never produced")
    # The calculator used to let you set 1 timeout left two minutes into the first quarter and
    # answered 45.1% — extrapolated from fourth quarters, measured nowhere. The slider floor is
    # now what has actually been observed by that point of a game.
    def timeout_range(app):
        """Find the home-timeout slider by label.

        These sliders deliberately carry no `key` — a keyed widget stores its value in session
        state, which would override the contextual defaults and leave them stuck — so the label
        is the only stable handle. Asserting there is exactly one makes a future collision fail
        here rather than silently picking the wrong control.
        """
        hits = [sl for sl in app.slider if sl.label == "Home timeouts left"]
        assert len(hits) == 1, f"expected one 'Home timeouts left' slider, found {len(hits)}"
        return hits[0].min, hits[0].max

    def timeout_floor(app):
        return timeout_range(app)[0]

    at.selectbox(key="period").set_value(1).run()
    at.select_slider(key="clock").set_value(600).run()      # 2 minutes into Q1
    floor_early = timeout_floor(at)
    check("early Q1 timeout floor is high", floor_early is not None and floor_early >= 5,
          f"floor was {floor_early}")

    at.selectbox(key="period").set_value(4).run()
    at.select_slider(key="clock").set_value(120).run()      # Q4, 2:00 left
    floor_late = timeout_floor(at)
    check("late Q4 allows zero timeouts", floor_late == 0, f"floor was {floor_late}")
    check("the floor falls as the game goes on", floor_early > floor_late,
          f"{floor_early} then {floor_late}")

    # The ceiling too: there is no forfeit rule, so seven stays reachable most of the game, but
    # nobody has held seven inside the last six minutes. One game in 6,135 carried all seven
    # into the fourth (MIN at UTA, 2021-12-31), so the bound must not rule that out either.
    _, ceil_late = timeout_range(at)                     # still Q4 2:00 from above
    check("late Q4 caps below seven", ceil_late == 6, f"ceiling was {ceil_late}")
    at.select_slider(key="clock").set_value(600).run()   # Q4 10:00
    _, ceil_early_q4 = timeout_range(at)
    check("seven still reachable earlier in the fourth", ceil_early_q4 == 7,
          f"ceiling was {ceil_early_q4}")

    print("\n9. The game curve has one point per second (hover cannot double up)")
    # A hover label printed twice, one on top of the other, because the selection keys on
    # elapsed time and up to 17 feed events can share a single second. Unique x values are the
    # fix, and this is the invariant that guarantees it.
    import pandas as pd
    from app import one_row_per_second
    from config import DATA_PROCESSED
    from train_model import TEST_SEASON

    raw = pd.read_parquet(DATA_PROCESSED / f"features_{TEST_SEASON}.parquet")
    dup_x = mono_break = final_changed = 0
    sample = raw.game_id.drop_duplicates().head(40)
    for gid in sample:
        g = raw[raw.game_id == gid]
        want_final = (int(g.score_home.max()), int(g.score_away.max()))
        k = one_row_per_second(g)
        if k["_elapsed"].duplicated().any():
            dup_x += 1
        if (k.score_home.diff() < 0).any() or (k.score_away.diff() < 0).any():
            mono_break += 1
        if (int(k.score_home.iloc[-1]), int(k.score_away.iloc[-1])) != want_final:
            final_changed += 1

    check(f"no duplicate x values in {len(sample)} games", dup_x == 0, f"{dup_x} games")
    check("score never goes backwards on the curve", mono_break == 0, f"{mono_break} games")
    check("final score is preserved", final_changed == 0, f"{final_changed} games")

    print("\n10. Every period selectable, including overtime")
    for label in ("Q1", "Q4", "OT1", "OT2"):
        at.selectbox(key="period").set_value(int(label[-1]) if label.startswith("Q") else 4 + int(label[-1])).run()
        check(f"period {label} renders", not at.exception, exceptions(at))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        sys.exit(1)
    print("All dashboard checks passed.")


if __name__ == "__main__":
    main()
