# Cavs Win Probability Model

Given any in-game state (score, time remaining, etc.), estimate the probability the home team wins.

## Setup (from a fresh environment)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

`requirements.txt` holds only what the deployed dashboard imports; `requirements-dev.txt` adds
`nba_api` and `lightgbm`, which the data pull and the model comparison need. They are kept
separate because putting the full set on the deploy host broke the first build.

## Run everything, in order

Every step caches its output, so a re-run skips work that is already done. The full data pull is
the only slow part — about 20 minutes per season, and it is resumable.

```bash
python src/data_pull.py               # 1. pull play-by-play from the NBA API (slow, cached)
python src/pull_venues.py             # 2. arenas, to identify neutral-site games
python src/identify_neutral_sites.py  # 3. flag games with no true home team
python src/build_game_states.py       # 4. reconstruct score/clock/possession at every event
python src/features.py                # 5. build the model inputs
python src/train_model.py             # 6. fit both models, write models/
python src/build_timeout_bounds.py    # 7. bounds so the tool cannot be asked impossible states
python src/fetch_logos.py             # 8. cache team logos locally for the dashboard
python src/validate.py                # 9. evaluate on the held-out season
```

Then check it, and open it:

```bash
python tests/test_predict.py               # model + prediction path
python tests/test_dashboard.py             # the dashboard renders and behaves
python tests/audit_calibration.py          # calibration across 81 game situations
streamlit run dashboard/app.py             # then open http://localhost:8501
```

Those four run straight from a clone — the processed features and the trained models are
committed. One more check needs the raw feed, which is too large to commit, so it requires
step 1 of the pipeline above to have been run first:

```bash
python tests/test_foul_timeout_accuracy.py # fouls/timeouts vs the NBA's own numbering
```

Optional:

```bash
python src/experiments.py             # configuration search, scored on validation only
```

## Phase 1: Data Pull

Pulls 5 regular seasons (2021-22 through 2025-26) of play-by-play data from the public
NBA Stats API via `nba_api`, and caches each season to parquet in `data/raw/`.

```bash
python src/data_pull.py
```

- Rate-limited to one call per 0.6s (parallelization isn't supported by the API).
- Retries transient failures with exponential backoff (2s, 4s, 8s, 16s).
- Resumable: a completed season is skipped; an interrupted season resumes from its last
  checkpoint (saved every 25 games) instead of re-pulling from scratch.
- Game IDs are kept as strings throughout — parquet preserves leading zeros; CSV/Excel would not.

Output per season in `data/raw/`:
- `games_{season}.parquet` — one row per game, with home/away teams and final outcome.
- `pbp_{season}.parquet` — one row per play-by-play event across every game in the season.

### Known data quirks found during validation

- `actionType` values are human-readable strings (`"Made Shot"`, `"Missed Shot"`, `"Foul"`, ...),
  not the lowercase categories (`2pt`, `3pt`, `foul`, ...) described in the API reference doc.
  Shot value (2 vs. 3 points) comes from the `shotValue` column, not from `actionType`.
- `scoreHome` / `scoreAway` are only populated on scoring plays and period markers — blank on
  most rows (rebounds, subs, fouls, missed shots, turnovers). Phase 2 forward-fills the last
  known score across each game to get the score at every moment.
- Team-attributed events (e.g. team rebounds) carry the team's ID in `personId`, not `0` as
  the "non-player event" documentation implies — only period/game markers are actually `0`.
- `MATCHUP` is normally differentiated across a game's two rows ("CLE vs. BOS" on the home
  team's row, "BOS @ CLE" on the away team's). For a handful of games per season it is
  **identical on both rows**, so home/away cannot be inferred from which row says "vs." —
  `parse_matchup()` reads the team order out of the string instead. The `ambiguous_matchup`
  column records where this occurred.
## Phase 2: Game States

```bash
python src/build_game_states.py   # -> data/interim/game_states_{season}.parquet
```

Converts the raw event feed into one row per moment: score, time remaining, possession,
and the label (`home_win`).

**What the numbers refer to**, since they are easy to conflate. The four training seasons
(2021-22 → 2024-25) total 4,910 games, not 4,174 — the difference is a 15% validation slice
(`VALIDATION_FRACTION` in `train_model.py`, 736 of 4,910 games) carved out of those same four
seasons, used only to measure which settings (like the leverage weight) scored best, never to
train the model itself:

| | Seasons | Games | Moments |
|---|---|---|---|
| Play-by-play pulled | 5 (2021-22 → 2025-26) | 6,135 | 3,011,185 |
| Four training seasons, before the split | 4 (2021-22 → 2024-25) | 4,910 | — |
| ↳ **Actually trained on** | — | **4,174** | **2,036,962** |
| ↳ Validation slice (tuning only, never trained on) | — | **736** | — |
| Held out for testing | 1 (2025-26) | 1,225 | 616,057 |
| Elo seed (final scores only) | 1 (2020-21) | 1,080 | — |

4,174 + 736 + 1,225 = 6,135 — every game pulled is accounted for exactly once.

A "moment" is one row per play-by-play event — every shot, rebound, foul and timeout —
carrying the score, clock and possession at that instant. About 490 per game.

The 2020-21 season contributes **no play-by-play at all**. It supplies final scores so the Elo
ratings have a starting point, which is what stops early-season games from being blind to team
strength. It is not a sixth season of training data.

Three raw-feed behaviours drive the implementation, each established by inspection:

- **`actionNumber` is not chronological.** ~4,000 rows a season sit out of order, with a
  median backwards clock jump of 75 seconds and a maximum of 720 (a full quarter). Rows are
  therefore ordered by game clock, with `actionNumber` breaking ties within an instant.
- **The running score is unreliable.** It appears on only ~26% of rows, and administrative
  rows carry stale or corrupt values — one period-end marker reads 119-112 after a
  buzzer-beater had made it 122-112; another game shows `0-1` in the closing seconds of a
  115-113 game. Because a basketball score can never decrease, forward-filling and then
  applying a running maximum repairs both. **This reconstructs the correct final score in
  all 6,150 games**, checked against `LeagueGameFinder`'s independent totals.
- **Possession is never stated.** It is inferred from the events that establish it (shots,
  turnovers, free throws, rebounds, and steals — which arrive as blank-`actionType` rows
  whose description contains "STEAL"). Unresolved on 0.4% of rows, all at game start before
  the first such event.

### Sanity check

Empirical home win rate by halftime margin, straight from the reconstructed states:

| Halftime margin | Games | Home win % |
|---|---|---|
| −20 or worse | 238 | 4.6% |
| −15 to −10 | 617 | 19.6% |
| −5 to 0 | 760 | 48.0% |
| Tied | 172 | 54.1% |
| 0 to +5 | 1076 | 63.9% |
| +10 to +15 | 620 | 86.1% |
| +20 or better | 300 | 98.7% |

Monotonic throughout, and a tied game at halftime lands at 54.1% — independently
reproducing NBA home-court advantage, which is evidence that score reconstruction and
home/away attribution are both correct.

## Phase 3: Features

```bash
python src/features.py   # -> data/processed/features_{season}.parquet
```

| Feature | Meaning |
|---|---|
| `score_margin` | Home score minus away score |
| `seconds_remaining` | Time left (in regulation; within OT, time left in that period) |
| `period`, `is_overtime` | Which period, and whether past regulation |
| `home_has_ball` | Who holds the ball, inferred from events |
| `home_fouls_period`, `away_fouls_period` | Team fouls committed this period |
| `home_in_bonus`, `away_in_bonus` | Whether the opponent has reached the 5th team foul |
| `home_timeouts_used`, `away_timeouts_used` | Timeouts taken so far |
| `momentum` | Change in margin over the last 3 minutes of game clock |
| `home_form`, `away_form`, `form_diff` | Average point differential over the prior 15 games |
| `margin_per_minute_left` | Margin scaled by time left to erase it |

**No look-ahead.** Every feature is knowable at that instant. `form` is the one with real
leakage risk and is shifted by one game, so a team's form entering a game excludes that game
and everything after it. Verified by recomputing a team's window by hand: game 16's form
equals the mean of games 1-15 exactly, and games 1-15 are left `NaN` rather than filled, so
"form unknown" stays its own case (81% of rows have form).

### Validation

Each feature was checked for real signal, not just for running without error.

*Pre-game form, measured at tip-off where the score is 0-0 and form is the only signal:*

| Form difference | Home win % |
|---|---|
| −33 to −8.9 | 26.6% |
| −4.1 to 0 | 53.6% |
| 0 to +3.9 | 57.4% |
| +8.9 to +29.9 | 78.8% |

*Possession, by how much time is left:*

| Situation | Home without ball | Home with ball |
|---|---|---|
| Tied, under 30s | 49.6% | 63.8% |
| Within 3, under 60s | 47.4% | 54.6% |
| Within 5, under 2min | 49.3% | 53.5% |

The effect shrinks as time remaining grows, which is the expected shape.

*Bonus threshold*, established from the data rather than assumed: the share of non-shooting
fouls producing free throws jumps from 15% at the 4th team foul to 68% at the 5th, 91% at
the 6th, 97%+ after. Hence `BONUS_THRESHOLD = 5`.

### Known caveats

- **Momentum carries little signal.** Holding margin and time roughly fixed, home win rate
  moves only between 51.6% and 57.1% across the full range of 3-minute swings, and not
  monotonically. Consistent with the general finding that scoring runs add little once score
  and time are controlled. Retained so the model can weigh it, but not expected to matter.
- **Timeouts used is confounded, not causal.** Home teams that have used 7 timeouts in a
  close 4th quarter win 35.6%, versus 59.5% for those who have used 4. This reflects trailing
  teams calling more timeouts, not timeouts causing losses. Legitimate information at that
  moment, but it must not be described as a causal effect.
- **Timeout allocation is unresolved.** Five regulation games show a team using 8 timeouts,
  which exceeds the expected allowance, so `timeouts_remaining` is deliberately not computed;
  only the directly observed `timeouts_used` is provided.

## Phase 4: Model Training and Validation

```bash
python src/train_model.py   # fits both models -> models/
python src/validate.py      # evaluation on the held-out season
python src/experiments.py   # configuration search, scored on validation only
```

**Splitting.** Whole games, never rows — two moments from the same game are near-duplicates
sharing an outcome, so a row-level split would let the model be graded on games it had
effectively already seen. The 2025-26 season is held out entirely by time: the model is fit
on 2021-22 through 2024-25 and never sees 2025-26 during training or configuration search.
That mirrors real use (fit on history, apply to a season that hasn't happened) and is a
harder test than a random split.

### Results on the held-out 2025-26 season (1,225 games, 616,057 states)

| Model | Log loss | Brier | AUC | Calibration gap |
|---|---|---|---|---|
| Always predict 55.3% | 0.6872 | 0.2470 | 0.500 | — |
| **Logistic regression** | **0.4411** | **0.1470** | **0.868** | **0.0159** |
| Gradient boosted (LightGBM) | 0.4441 | 0.1485 | 0.865 | 0.0159 |

### Pre-game team strength: Elo, not point differential

Team strength is an **Elo rating** (`src/elo.py`), not a trailing point differential. The
differential version was replaced because it is blind to schedule — +6 per game against a soft
slate and +6 against contenders score identically — which left tip-off estimates clustered near
the home-court baseline no matter who was playing.

Elo prices each result against the opponent's rating, so the same win is worth more against a
strong team. Margin of victory counts with diminishing returns, damped for teams already
rated highly. Ratings carry across seasons, regressed a quarter of the way toward the mean each
offseason, and are seeded by one extra prior season of final scores so that early-season games
are not strength-blind.

Home advantage is set to **37 Elo points**, fitted to the 55.3% home win rate in this data —
not the commonly quoted 100, which implies 64% and belongs to an earlier era.

Elo alone, before a single possession is played, picks winners at **65.8%** across 6,150 games
(log loss 0.6194, against 0.6875 for a constant). Pre-game accuracy has a low ceiling in any
case: almost everything that decides a basketball game has not happened yet at tip-off.

**Margin of victory is counted, and that was tested rather than assumed.** Weighting margin
means a team can rate above its record — Charlotte finished 2025-26 at 44-38 but rated 6th,
having gone 18-4 in games decided by 20+ and 5-11 in games decided by five or fewer. That looks
wrong to a basketball eye, so both versions were compared on the cleanest available test: take
each team's rating at the halfway point of a season, and see which better predicts what that
team actually does in the *second* half.

| Elo version | Predicts 2nd-half win % | Predicts 2nd-half point differential |
|---|---|---|
| **Margin-aware (kept)** | **0.729** | **0.733** |
| Wins only | 0.715 | 0.706 |

Margin-aware wins on both, so it stays. The rating is right that Charlotte were a better team
than their record; they converted it badly in close games.

Swapping it in improved every measure on the held-out season: log loss 0.4551 → 0.4411, Brier
0.1530 → 0.1470, AUC 0.8568 → 0.8683.

**The logistic regression is the primary model.** It was expected to be the baseline and it
won on every measure. That is not a tuning fluke: win probability is a smooth monotonic
function of margin and time, which is the shape logistic regression represents natively,
while trees approximate a smooth surface with axis-aligned steps. `margin_per_minute_left`
supplies the one interaction that would otherwise favor trees. Smoothness also matters
practically — a tree ensemble draws a visibly jagged win probability curve across a game.

### Real situations, model against what actually happened

Rates and intervals are computed **per game**, not per state: many states come from the same
game and share its outcome, so treating them as independent overstates the sample severalfold
and makes noise look like model error.

| Situation | Games | Actually won | Model says |
|---|---|---|---|
| Tied, under 30s, home has ball | 66 | 68.2% ±11 | 58.9% |
| Tied, under 30s, home does NOT have ball | 64 | 53.1% ±12 | 44.5% |
| Home down 3, under 15s, home ball | 50 | 8.0% ±8 | 3.3% |
| Home up 5, under 60s, home ball | 88 | 98.9% ±2 | 99.2% |
| Home up 2, under 60s, away ball | 84 | 86.9% ±7 | 86.1% |
| Within 3, start of 4th | 270 | 51.5% ±6 | 51.6% |
| Tied entering overtime | 54 | 61.1% ±13 | 54.1% |
| Home down 10, start of 4th | 105 | 22.9% ±8 | 15.6% |

Two entries look like large model misses — ~10 points low on "tied with the ball late" and
~8 low on "down 10 entering the 4th". **They are not.** Measuring the same situations in
every season shows how much they move year to year:

| Situation | 21-22 | 22-23 | 23-24 | 24-25 | 25-26 | Range |
|---|---|---|---|---|---|---|
| Tied, under 30s, home has ball | 60.9 | 60.7 | 64.6 | 52.9 | **68.2** | 52.9 – 68.2 |
| Home down 10, start of 4th | 18.8 | 17.8 | 13.4 | 15.7 | **22.9** | 13.4 – 22.9 |
| Tied entering overtime | 48.3 | 54.4 | 64.4 | 51.7 | **61.1** | 48.3 – 64.4 |

The test season is the **highest of all five** on each. Any one season holds only 50-100
games of a given tight situation, so its rate carries an interval near ±10 points, and the
genuine year-to-year spread is 10-15 points wide. The model's ~58.6% and ~15.3% sit mid-range.

Pooling all five seasons gives the stable estimates: 61.4% ±5.1 (352 games) and 17.6% ±3.3
(523 games). Against those, the model is within about 3 and 2 points respectively — inside
the sampling interval in both cases.

**So: no material underconfidence.** An earlier version of this README claimed a 7-10 point
endgame weakness; that was one season's high draw being read as a model defect. `validate.py`
now prints the per-season table alongside the test-season comparison so the same misreading
cannot recur.

### Leverage weighting

Close, late states are upweighted during training (`LEVERAGE_WEIGHT_STRENGTH = 15`), chosen
on the validation split with the test season untouched. Without it the model was close to
useless where it matters most: endgame states are a fraction of a percent of three million
rows, so getting them right barely moves average log loss and the model didn't bother. It
predicted a **3.3-point** gap between having the ball and not in a tied game under 30
seconds, where the real gap is about 15. Weighting lifted that to ~10 points **and** improved
log loss both overall and on late close states — accuracy and realism moved together rather
than trading off.

### Canonical game states

States whose answer basketball already knows, checked against what actually happened. These are
not arithmetic tests — a wrong sign or a mis-scaled feature shows up here as a number a coach
would laugh at. Observed rates count **one vote per game** across all five seasons, from the
matching band in each row. Run them with `python tests/test_predict.py`.

| State | Model says | Actually won | Games |
|---|---|---|---|
| +20, 1:00 left in Q4 | 99.9% | 100.0% | 376 |
| −20, 1:00 left in Q4 | 0.1% | 0.0% | 262 |
| +10, 5:00 left in Q4 | 92.9% | 95.8% | 571 |
| +5, 2:00 left in Q4 | 88.7% | 90.4% | 684 |
| Tied at tip-off | 53.5% | 55.3% | 6,135 |
| Tied, under 0:30, home ball | 63.3% | 61.4% | 352 |
| Tied, under 0:30, away ball | 49.9% | 46.6% | 356 |

Three orderings are asserted separately, and must hold whatever the numbers do: a lead beats its
mirror-image deficit; a lead and its mirror sum to about 1; and the same lead is worth more late
than early.

The first row is the worked example the assessment itself gives — +20 with a minute left should
read about 99%. It reads 99.9%, against 376 games in that position of which every one was won.

### Known calibration bias: the model is too confident in the leader

`tests/audit_calibration.py` sweeps 81 buckets — nine score bands across nine clock windows —
counting **one vote per game** so that repeated states from the same game cannot inflate the
sample. Wilson intervals give the range the true rate could plausibly occupy at each bucket's
size, which is what separates a real miss from a small-sample one.

On average the model is well calibrated: the game-weighted miss is **1.69 percentage points of
win probability**, and **77 of 81** buckets land within 5 points of the observed rate. But **28
buckets miss by more than sampling error can explain**, and they lean almost entirely one way.

| Situation | Model says | Actually won | Games |
|---|---|---|---|
| Q1, home up 8-14 | 76.3% | **70.9%** | 2,341 |
| Q1, home down 8-14 | 31.4% | **37.6%** | 2,082 |
| Q2, home up 8-14 | 78.7% | **75.6%** | 2,847 |
| Q4 under 0:30, home up 1-3 | 90.5% | **84.5%** | 849 |
| Q4 under 0:30, home down 1-3 | 12.6% | **16.4%** | 858 |

The direction is consistent: **whoever leads is given more credit than the record supports, and
whoever trails is given less.** Put in basketball terms, comebacks happen more often than this
model expects. The effect is largest early — a first-quarter lead is treated as more durable
than five seasons of games say it is.

The bias is real rather than noise: it shows up in the held-out season alone (15 of 81 buckets)
and again across all five (28 of 81). It is also small in absolute terms, a few points in the
affected buckets against an average miss under two, and it does not move the headline metrics.

**It is documented rather than patched.** A post-hoc correction fitted to these buckets could
not then be validated, because the held-out season is the same data measuring the problem —
tuning on it would forfeit the only clean test available. The honest summary is that the model
is well calibrated on average and overstates leads at the margin, and that a coach reading a
first-quarter lead of 8-14 should shade it down by roughly five points.

## Phase 5: Dashboard

```bash
streamlit run dashboard/app.py     # then open http://localhost:8501
python tests/test_predict.py       # verifies the dashboard predicts identically to the pipeline
```

**Game replay** — pick a real 2025-26 game (a season the model never trained on) and watch
the win probability curve, with a table of the plays that moved it most. Nothing is entered
by hand; every value comes from the game.

**Situation calculator** — four controls (margin, period, clock, possession), with team
strength, fouls and timeouts behind a "more options" panel. Momentum is deliberately absent:
it was measured, it barely moves the prediction, and a control that does nothing implies an
effect the data does not support.

### One prediction path

`src/predict.py` is the only way anything outside training produces a number, and
`tests/test_predict.py` asserts that a state rebuilt through it reproduces the pipeline's own
stored features and predictions exactly (all 19 feature columns, both models). Without that,
any drift between how the dashboard and the training code build features would produce
confident, plausible, wrong numbers with nothing raising an error.

### Bugs this phase surfaced

Three real defects, all found by running the thing rather than reading the code:

1. **Overtime sorted inside the fourth quarter.** The elapsed-time helper counted completed
   periods as `min(period, 4) - 1`, which credits an overtime period with only three finished
   quarters — so OT began at 2160s, exactly where the 4th began. The game selector displayed
   a final score of "104-104", an impossible result, which is what exposed it. This corrupted
   row order, momentum, and foul/timeout alignment in all **310 overtime games**. Fixed to
   `min(period - 1, 4)`; all 6,135 games now end on their true final score.
2. **A one-point lead at 0:00 was called 74.5%.** The game is over at that point — but no
   smooth function of margin and time can say so. Two features were added: margin scaled by
   *sqrt* of time left (scores accumulate like a random walk, so spread grows with the square
   root of time, not time), and an explicit "decided" indicator. That reads 97.7% now.
3. **Flat defaults skewed the calculator.** With only score, time and possession supplied, the
   rest were filled with fixed league averages — including two timeouts used, which is normal
   in the first quarter and wrong with two minutes left in the fourth, where the median is
   five. That inflated a tied-game-with-the-ball reading from a realistic ~51% to 63%.
   Defaults are now conditioned on where in the game you are, and a regression test pins the
   calculator to within 5 points of what the model gives for real states in the same spot.

## Phase 5 scope

Two views:

1. **Game replay** — pick a real game, watch the win probability curve unfold. No inputs;
   every feature comes from the game's own data. This is the view that demonstrates the
   model is trustworthy, since the curve can be read against what actually happened.
2. **Scenario calculator** — four controls: score margin, time remaining, period, and
   **possession**. Possession is included on evidence, not instinct: in a tied game under 30
   seconds it is worth ~14 percentage points, and without it the tool would return a single
   averaged number for "down 2 with 18 seconds" that is wrong whether or not you have the
   ball. Team strength, bonus state and timeouts sit behind a collapsed section, defaulted
   to neutral. Momentum is deliberately *not* a control — it was measured and barely moves
   the prediction, so exposing it would imply an effect the data does not support.

## Venue data and neutral-site games

```bash
python src/pull_venues.py            # ScoreboardV2, one call per game date (~9 min)
python src/identify_neutral_sites.py # flags -> data/interim/game_flags_{season}.parquet
```

`ScoreboardV2` supplies `ARENA_NAME` plus the NBA's own `HOME_TEAM_ID`. It is queried per
date (~165 dates a season) rather than per game, so it is far cheaper than a per-game
endpoint. This serves two purposes:

**Home/away verification.** Our `MATCHUP`-derived home team was checked against the NBA's
`HOME_TEAM_ID` for all 6,148 games with venue data. Zero disagreements.

**Neutral-site detection**, done from the data rather than from a hardcoded list of cities,
so it catches international games, NBA Cup games in Las Vegas, alternate home venues, and
one-off relocations alike. Two conditions must both hold:

1. The game is at a different arena than the home team's primary arena, *and its date falls
   inside* that primary arena's date range. Arena renames are permanent, so old and new
   names occupy separate stretches of the calendar; a neutral-site game interrupts a run of
   normal home dates.
2. That arena hosts at most `MAX_NEUTRAL_GAMES_PER_ARENA` (4) of the team's home games.

Both are needed. Name comparison alone flags every renamed building — Staples Center became
Crypto.com Arena mid-2021-22, which alone produced 39 false positives. The date test alone
still misses messy transitions: Phoenix's 2025-26 games alternate between "PHX Arena" and
"Mortgage Matchup Center" for weeks. Together they yield **22 neutral-site games across five
seasons** (0.36% of games): Mexico City, Paris, Berlin, London, NBA Cup semifinals in Las
Vegas, and the Spurs' games at the Alamodome and Austin's Moody Center.

Two 2025-26 games (`0022500369`, `0022500370`) are absent from ScoreboardV2 on every nearby
date and carry `venue_known = False`; they are neither confirmed nor excluded.

- **`ambiguous_matchup` is not a neutral-site flag.** In 2024-25 and 2025-26 every affected
  game happens to be neutral-site (Mexico City, Paris, NBA Cup semifinals), but neutral-site
  games in earlier seasons (2023-24 Cup semifinals in Las Vegas, the 2024 Paris Game) return
  normal differentiated matchups and are not flagged. Genuinely identifying neutral-site
  games would need a venue source; see the limitations section of the writeup.
