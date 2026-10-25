# V6 Prelaunch Summary

- Status: PRELAUNCH_READY_NOT_STARTED
- Purpose: targeted short ablation of PreCap cap threshold c and soft-limit width s after v5 depth-64 diagnosis.
- Why now: v5 suggested no simple gradient crash; instead seed-, task-, and reference-dependent late dynamics at d64.
- Core question: is the d64 weakness better explained by over-limiting, under-limiting, or task-specific interaction?

## Criteria checked

- Launcher present and executable.
- Config count = 105.
- Frozen baseline formula matches exactly: formula_max_abs_err = 0.0.
- Cap branch continuity jump is numerically tiny.
- Prelaunch status clean: 0/105 full runs, 0/630 checkpoint rows, done = False.

## Design

- Cells: radial_ripple d64, ridge_mix d64, radial_ripple d16.
- Seeds: 900-904.
- Acts: precap_c075, precap_base, precap_c125, precap_s075, precap_s225, silu, gelu.
- Steps: 1200 with checkpoints 0, 100, 300, 500, 800, 1200.
- Workers: 4.

## Decision rule

At least one ablation variant must beat precap_base at d64 with lower test MSE and at least 4/5 favorable paired seeds in a target cell to count as meaningful evidence.
