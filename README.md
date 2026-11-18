# GTA VI launch-day Twitch viewer forecast

A quantile forecaster for the `gta6-twitch-ccu-forecast-01` bounty: given everything Twitch's
"Grand Theft Auto VI" category shows up to the end of the first UTC hour of 2026-11-19, it answers
with five quantiles of the category's concurrent viewers for each of hours 2 to 24.

[![bounty](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/flavyn/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/flavyn/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

## How it forecasts

The score is the mean pinball loss on the log scale, so the audience's absolute size cancels out:
what matters is the shape of the day as ratios to hour 1, and an honest spread around that shape.

1. Every comparable launch in the case file becomes a 24-slot profile of mean viewers per hour of
   its own release day.
2. Each remaining comparable contributes `r = ln(profile[h-1] / profile[0])` for hour `h`, weighted
   by recency (two-year half-life) and by how close its hour-1 level was to this case's.
3. The weighted median of those `r` values is the centre of hour `h`; a weighted MAD around it,
   scaled by 0.7, is the log-scale spread.
4. `q_tau(h) = p1 * exp(centre + z_tau * spread)` for the five scored levels, rounded and forced
   non-decreasing.

A fixed median shape stands in if a case leaves too few comparable observations for an hour.

## Run

```bash
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output node main.ts
```

Node 22, standard library only, no network. Reads every `*.in` in the input directory in name order
and writes `<id>.out` beside the outputs directory; two runs produce identical bytes.
