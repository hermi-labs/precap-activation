# v9_depth — depth stress test

## Purpose
After v9, run a focused depth stress test before the broad v9b sweep.

## Idea
Take the strongest current candidates and ask one question: do they remain stable and useful when depth doubles from 64 to 128?

## Candidates
- SiLU (fair plain-depth reference moved here as residual depth reference)
- GELU
- CapExp(c=1.0)
- PreCap(c=0.5,a=1.5,gamma=0,s=1.5)
- PreCap(c=1.0,a=1.0,gamma=0.01,s=1.5)

## Grid
3 targets × 2 depths (64,128) × res_inv only × 5 candidates × 10 seeds = 300 runs.

## Metrics
Same checkpoints and diagnostics as v9: loss curves, gradient norm, activation std, activation max, cap fraction.

## Why before v9b
This is a cheap gate test for the “engine + controller” idea: if the best candidates do not survive 128 layers under residual scaling, the large v9b sweep should be adjusted before burning time.

## Start after v9
`caffeinate -i bash /Users/aaronschmitt/PyCharmMiscProject/v9_depth/run_v9_depth.sh`
