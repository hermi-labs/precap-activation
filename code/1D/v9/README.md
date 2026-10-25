# v9 — Instrumented confirmation run

## Purpose
Confirm the v8 winners with 10 seeds and record learning curves plus layer diagnostics.

## Families
- CapExp: c=1.0 and 1.5
- PreCap: two v8-derived candidates
- SiLU, GELU, ReLU baselines

## Grid
3 targets × depths 32/64 × plain/res_inv × 7 candidates × 10 seeds = 840 runs.

## Logged metrics
At steps 1, 2, 5, 10, 20, 50, 100, 200, 400, 800, 1600 and 3000:
- train/test loss
- global gradient norm
- mean layer activation standard deviation
- maximum activation magnitude
- fraction in the CapExp linear regime

## Start
`caffeinate -i bash /Users/aaronschmitt/PyCharmMiscProject/v9/run_v9.sh`
