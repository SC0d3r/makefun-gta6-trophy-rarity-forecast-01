# GTA VI trophy rarity forecast

Entry for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty:
https://make.fun/bounty/gta6-trophy-rarity-forecast-01

The program reads a case of Exophase history and writes five quantiles (5, 25, 50, 75, 95) for every
trophy's earn rate at the target. Reads with the same page state are collapsed, null rates are
carried forward, and hours since launch anchor the timeline.

Players are projected from recent growth with saturation and log-normal uncertainty. Rates blend
a saturating logistic curve with description-based ceilings and an earners/players decomposition,
in logit space, plus persistence for very rare trophies. A story-chain correction orders the
chapter trophies and constrains the platinum and prologue rows. A deterministic 1500-draw
xorshift Monte Carlo carries player, timing, ceiling, slope, marginal-completion and Poisson
uncertainty into the printed quantiles.

Run from this repository:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
Node 22, standard library only, no network. Every trophy and case has a house baseline fallback.
