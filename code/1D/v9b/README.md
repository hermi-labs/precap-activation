# v9b — Fairness + steering sweep

## Purpose
Comprehensive confirmation run that incorporates the v8/v9 lessons and fairly tests the motor plus steering controller.

## What it adds
- Fair init sweep for all baselines (init_k = 0.5 and 1.0)
- Broader CapExp steering sweep (c = 0.75, 1.0, 1.25, 1.5; init_k = 0.25, 0.5, 1.0)
- Broader PreCap sweep around the best v8 settings
- Depths 16, 32, 64
- Plain and res_inv
- 10 seeds
- Same learning-curve and layer diagnostics as v9

## Grid size
6480 runs total.

## Logged checkpoints
1, 2, 5, 10, 20, 50, 100, 200, 400, 800, 1600, 3000.

## Goal
Measure not just final loss, but stable operating region: low loss, high tolerable learning rate, depth robustness, and low failure/kollaps rate.

## Start after v9
`caffeinate -i bash /Users/aaronschmitt/PyCharmMiscProject/v9b/run_v9b.sh`
