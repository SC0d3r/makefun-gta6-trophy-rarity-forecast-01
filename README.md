# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/SC0d3r/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/SC0d3r/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A quantile forecaster for GTA VI trophy earn rates, written twice with the same
model: `main.ts` for Node 22 and `main.py` for Python 3.13. Both use only the
standard library.

The model first collapses consecutive identical reads into one distinct page,
so a stalled recorder cannot look like new evidence. Each trophy is then moved
to the logit scale, y = ln(p / (1 - p)), with p clamped to 0.01..99.99. A
recency-weighted robust line through the distinct refreshes gives a per-day
slope when at least three points exist; short or noisy fits are pulled toward
zero and toward the median slope of the trophy's tier. The horizon move is
damped smoothly and capped at 1.5 logit units, and the median forecast is the
last observed y plus 0.75 of that move. A robust scale of one-refresh-ahead
changes per square-root day, with a pooled tier or global fallback, sets the
spread together with a 0.10 logit refresh-timing term. Normal quantiles on that
scale become the five output probabilities, rounded to four decimals and forced
to be finite, in range and non-decreasing. A trailing null rate uses the
trophy's nearest earlier value, and rates at 0 or 100 produce literal boundary
rows.

Run it from this directory:

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

The Python twin takes the same variables:

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`MAKEFUN_INPUT_DIR` defaults to `/input` and `MAKEFUN_OUTPUT_DIR` defaults to
`/output`. The program reads every `*.in` case and writes one `<id>.out` file
per case.

The program is fully offline: it uses no network, no packages and no data
beyond the case files it is given.
