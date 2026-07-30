# NBA Win Probability — Live Dashboard

Given any in-game state (score, time remaining, possession, and more), this estimates the
probability that the home team wins.

**This repository hosts the interactive dashboard only.** The full project — the data pipeline,
feature engineering, model training, validation suite, written methodology and AI-usage
disclosure — is provided separately to the reviewer.

## Running it locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## What the model is

A logistic regression over 25 columns, built from the score margin, time remaining, period,
possession, each team's fouls and bonus status, each team's timeouts left, momentum over the
last three minutes, and each team's Elo strength coming in — plus combinations of score against
the clock, which carry most of the model's weight.

Trained on five seasons of public NBA play-by-play (2021-22 to 2024-25) and tested on 2025-26,
which was held back entirely and never seen during training. The split is by whole game, never
by individual event.

| Held-out season (1,225 games) | Model | Guessing the league home-win rate |
|---|---|---|
| Log loss | **0.4370** | 0.6872 |
| Brier score | **0.1454** | 0.2470 |
| AUC | **0.871** | 0.500 |
| Calibration | **within 1.6 points** | — |

Sanity checks against situations whose answers are already known: 20 points up with a minute
left reads over 99.9%, and teams in that position won 539 of 539; 20 down reads under 0.1% and
won none of 387.

## The two tabs

**Win Probability** — set a situation and read the home team's chance of winning from there.

**Test It Yourself** — pick any real game in the data and see the model's curve against what
actually happened, so the number can be checked rather than taken on trust.

## Data

The public NBA Stats API, via the `nba_api` Python library. No betting data, no third-party
ratings, no scraped sites.

Game states are committed under `data/processed/` so the app runs immediately. The raw pull is
excluded from this repository and is reproducible from the full project.
