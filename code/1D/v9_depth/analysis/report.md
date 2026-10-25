# v9_depth Auswertung

Runs: 300 (Ausfaelle: 41); Ausfaelle zaehlen als unendlicher Loss.

## Median Test-Loss, Ausfaelle, Rang

### abs_wave, Tiefe 64

| Kandidat | Median | IQR | 95%-CI | Ausfaelle | Rang |
|---|---|---|---|---|---|
| SiLU | 2.75e-04 | 2.17e-04 - 3.20e-04 | 1.94e-04 - 3.34e-04 | 0/10 | 3 |
| GELU | 5.11e-04 | 4.19e-04 - 1.20e-03 | 4.08e-04 - 1.21e-03 | 0/10 | 4 |
| CapExp (c=1.0) | FAIL | 4.67e-03 - FAIL | 1.17e-03 - FAIL | 5/10 | 5 |
| PreCap A (c=0.5) | 7.94e-05 | 3.91e-05 - 1.22e-04 | 3.49e-05 - 1.98e-04 | 0/10 | 2 |
| PreCap B (c=1.0) | 5.86e-05 | 4.87e-05 - 7.50e-05 | 4.50e-05 - 7.60e-05 | 0/10 | 1 |

### abs_wave, Tiefe 128

| Kandidat | Median | IQR | 95%-CI | Ausfaelle | Rang |
|---|---|---|---|---|---|
| SiLU | 2.56e-04 | 1.70e-04 - 4.49e-04 | 1.48e-04 - 4.51e-04 | 0/10 | 3 |
| GELU | 1.10e-03 | 8.64e-04 - 2.07e-03 | 6.90e-04 - 2.24e-03 | 0/10 | 4 |
| CapExp (c=1.0) | FAIL | FAIL - FAIL | FAIL - FAIL | 10/10 | 5 |
| PreCap A (c=0.5) | 1.44e-04 | 8.00e-05 - 5.28e-04 | 6.79e-05 - 5.52e-04 | 0/10 | 2 |
| PreCap B (c=1.0) | 7.50e-05 | 5.63e-05 - 1.70e-04 | 5.55e-05 - 3.34e-04 | 0/10 | 1 |

### sin3_quad, Tiefe 64

| Kandidat | Median | IQR | 95%-CI | Ausfaelle | Rang |
|---|---|---|---|---|---|
| SiLU | 4.72e-05 | 4.03e-05 - 5.04e-05 | 3.84e-05 - 6.29e-05 | 0/10 | 3 |
| GELU | 4.63e-04 | 2.30e-04 - 9.73e-04 | 1.96e-04 - 1.64e-03 | 0/10 | 5 |
| CapExp (c=1.0) | 2.08e-05 | 1.64e-05 - 7.03e-05 | 1.53e-05 - 8.57e-05 | 2/10 | 1 |
| PreCap A (c=0.5) | 8.49e-05 | 4.16e-05 - 1.25e-04 | 3.77e-05 - 2.07e-04 | 0/10 | 4 |
| PreCap B (c=1.0) | 3.53e-05 | 2.70e-05 - 5.68e-05 | 2.10e-05 - 1.58e-04 | 0/10 | 2 |

### sin3_quad, Tiefe 128

| Kandidat | Median | IQR | 95%-CI | Ausfaelle | Rang |
|---|---|---|---|---|---|
| SiLU | 3.26e-05 | 2.65e-05 - 7.53e-05 | 2.56e-05 - 1.10e-04 | 0/10 | 1 |
| GELU | 4.87e-04 | 3.10e-04 - 6.23e-04 | 2.83e-04 - 6.34e-04 | 0/10 | 4 |
| CapExp (c=1.0) | FAIL | FAIL - FAIL | FAIL - FAIL | 10/10 | 5 |
| PreCap A (c=0.5) | 1.19e-04 | 1.02e-04 - 2.68e-04 | 9.43e-05 - 2.91e-04 | 0/10 | 3 |
| PreCap B (c=1.0) | 7.09e-05 | 5.14e-05 - 9.97e-05 | 4.95e-05 - 1.06e-04 | 0/10 | 2 |

### sin_mix, Tiefe 64

| Kandidat | Median | IQR | 95%-CI | Ausfaelle | Rang |
|---|---|---|---|---|---|
| SiLU | 1.79e-03 | 1.62e-03 - 2.00e-03 | 1.57e-03 - 2.12e-03 | 0/10 | 5 |
| GELU | 1.32e-03 | 1.19e-03 - 1.43e-03 | 1.12e-03 - 2.27e-03 | 0/10 | 3 |
| CapExp (c=1.0) | 1.42e-03 | 9.06e-04 - FAIL | 8.71e-04 - FAIL | 4/10 | 4 |
| PreCap A (c=0.5) | 4.40e-04 | 3.99e-04 - 5.14e-04 | 3.96e-04 - 7.02e-04 | 0/10 | 1 |
| PreCap B (c=1.0) | 6.39e-04 | 4.94e-04 - 7.65e-04 | 4.91e-04 - 8.57e-04 | 0/10 | 2 |

### sin_mix, Tiefe 128

| Kandidat | Median | IQR | 95%-CI | Ausfaelle | Rang |
|---|---|---|---|---|---|
| SiLU | 1.83e-03 | 1.61e-03 - 2.11e-03 | 1.60e-03 - 2.15e-03 | 0/10 | 4 |
| GELU | 1.45e-03 | 1.13e-03 - 4.28e-03 | 1.07e-03 - 5.18e-03 | 0/10 | 3 |
| CapExp (c=1.0) | FAIL | FAIL - FAIL | FAIL - FAIL | 10/10 | 5 |
| PreCap A (c=0.5) | 3.65e-04 | 3.37e-04 - 5.15e-04 | 3.28e-04 - 5.23e-04 | 0/10 | 1 |
| PreCap B (c=1.0) | 5.43e-04 | 4.98e-04 - 6.18e-04 | 4.77e-04 - 8.40e-04 | 0/10 | 2 |

## Tiefenwirkung (Median 128 / Median 64)

| Ziel | Kandidat | 64 | 128 | Faktor |
|---|---|---|---|---|
| abs_wave | CapExp (c=1.0) | FAIL | FAIL | n/a |
| abs_wave | GELU | 5.11e-04 | 1.10e-03 | 2.15e+00 |
| abs_wave | PreCap A (c=0.5) | 7.94e-05 | 1.44e-04 | 1.82e+00 |
| abs_wave | PreCap B (c=1.0) | 5.86e-05 | 7.50e-05 | 1.28e+00 |
| abs_wave | SiLU | 2.75e-04 | 2.56e-04 | 9.31e-01 |
| sin3_quad | CapExp (c=1.0) | 2.08e-05 | FAIL | n/a |
| sin3_quad | GELU | 4.63e-04 | 4.87e-04 | 1.05e+00 |
| sin3_quad | PreCap A (c=0.5) | 8.49e-05 | 1.19e-04 | 1.40e+00 |
| sin3_quad | PreCap B (c=1.0) | 3.53e-05 | 7.09e-05 | 2.01e+00 |
| sin3_quad | SiLU | 4.72e-05 | 3.26e-05 | 6.91e-01 |
| sin_mix | CapExp (c=1.0) | 1.42e-03 | FAIL | n/a |
| sin_mix | GELU | 1.32e-03 | 1.45e-03 | 1.10e+00 |
| sin_mix | PreCap A (c=0.5) | 4.40e-04 | 3.65e-04 | 8.30e-01 |
| sin_mix | PreCap B (c=1.0) | 6.39e-04 | 5.43e-04 | 8.51e-01 |
| sin_mix | SiLU | 1.79e-03 | 1.83e-03 | 1.02e+00 |

## Signifikante Paarvergleiche (Mann-Whitney, Holm-korrigiert, p<0.05)

| Ziel | Tiefe | Besser | Schlechter | Median besser | Median schlechter | p_holm |
|---|---|---|---|---|---|---|
| abs_wave | 64 | PreCap B (c=1.0) | CapExp (c=1.0) | 5.86e-05 | FAIL | 7.8e-03 |
| abs_wave | 64 | PreCap B (c=1.0) | GELU | 5.86e-05 | 5.11e-04 | 8.6e-03 |
| abs_wave | 64 | PreCap B (c=1.0) | SiLU | 5.86e-05 | 2.75e-04 | 1.1e-02 |
| abs_wave | 64 | PreCap A (c=0.5) | CapExp (c=1.0) | 7.94e-05 | FAIL | 2.0e-02 |
| abs_wave | 64 | SiLU | GELU | 2.75e-04 | 5.11e-04 | 4.5e-02 |
| abs_wave | 128 | SiLU | CapExp (c=1.0) | 2.56e-04 | FAIL | 3.8e-03 |
| abs_wave | 128 | GELU | CapExp (c=1.0) | 1.10e-03 | FAIL | 3.8e-03 |
| abs_wave | 128 | PreCap A (c=0.5) | CapExp (c=1.0) | 1.44e-04 | FAIL | 3.8e-03 |
| abs_wave | 128 | PreCap B (c=1.0) | CapExp (c=1.0) | 7.50e-05 | FAIL | 3.8e-03 |
| abs_wave | 128 | PreCap B (c=1.0) | GELU | 7.50e-05 | 1.10e-03 | 2.2e-02 |
| abs_wave | 128 | PreCap A (c=0.5) | GELU | 1.44e-04 | 1.10e-03 | 3.6e-02 |
| abs_wave | 128 | SiLU | GELU | 2.56e-04 | 1.10e-03 | 4.5e-02 |
| sin3_quad | 64 | SiLU | GELU | 4.72e-05 | 4.63e-04 | 1.1e-02 |
| sin3_quad | 128 | SiLU | CapExp (c=1.0) | 3.26e-05 | FAIL | 3.8e-03 |
| sin3_quad | 128 | GELU | CapExp (c=1.0) | 4.87e-04 | FAIL | 3.8e-03 |
| sin3_quad | 128 | PreCap A (c=0.5) | CapExp (c=1.0) | 1.19e-04 | FAIL | 3.8e-03 |
| sin3_quad | 128 | PreCap B (c=1.0) | CapExp (c=1.0) | 7.09e-05 | FAIL | 3.8e-03 |
| sin3_quad | 128 | PreCap B (c=1.0) | GELU | 7.09e-05 | 4.87e-04 | 1.4e-02 |
| sin_mix | 64 | PreCap B (c=1.0) | SiLU | 6.39e-04 | 1.79e-03 | 1.4e-02 |
| sin_mix | 64 | PreCap A (c=0.5) | SiLU | 4.40e-04 | 1.79e-03 | 1.7e-02 |
| sin_mix | 64 | PreCap B (c=1.0) | GELU | 6.39e-04 | 1.32e-03 | 3.6e-02 |
| sin_mix | 128 | SiLU | CapExp (c=1.0) | 1.83e-03 | FAIL | 3.8e-03 |
| sin_mix | 128 | GELU | CapExp (c=1.0) | 1.45e-03 | FAIL | 3.8e-03 |
| sin_mix | 128 | PreCap A (c=0.5) | CapExp (c=1.0) | 3.65e-04 | FAIL | 3.8e-03 |
| sin_mix | 128 | PreCap B (c=1.0) | CapExp (c=1.0) | 5.43e-04 | FAIL | 3.8e-03 |
| sin_mix | 128 | PreCap A (c=0.5) | SiLU | 3.65e-04 | 1.83e-03 | 8.6e-03 |
| sin_mix | 128 | PreCap B (c=1.0) | SiLU | 5.43e-04 | 1.83e-03 | 8.6e-03 |
| sin_mix | 128 | PreCap A (c=0.5) | GELU | 3.65e-04 | 1.45e-03 | 8.6e-03 |
| sin_mix | 128 | PreCap B (c=1.0) | GELU | 5.43e-04 | 1.45e-03 | 4.5e-02 |

Nicht signifikant: 31 von 60 Vergleichen.

### Nicht signifikante Paare

| Ziel | Tiefe | A | B | p_holm |
|---|---|---|---|---|
| abs_wave | 64 | SiLU | CapExp (c=1.0) | 0.06 |
| abs_wave | 64 | SiLU | PreCap A (c=0.5) | 0.36 |
| abs_wave | 64 | GELU | CapExp (c=1.0) | 0.49 |
| abs_wave | 64 | GELU | PreCap A (c=0.5) | 0.06 |
| abs_wave | 64 | PreCap A (c=0.5) | PreCap B (c=1.0) | 1.00 |
| abs_wave | 128 | SiLU | PreCap A (c=0.5) | 1.00 |
| abs_wave | 128 | SiLU | PreCap B (c=1.0) | 0.72 |
| abs_wave | 128 | PreCap A (c=0.5) | PreCap B (c=1.0) | 1.00 |
| sin3_quad | 64 | SiLU | CapExp (c=1.0) | 1.00 |
| sin3_quad | 64 | SiLU | PreCap A (c=0.5) | 1.00 |
| sin3_quad | 64 | SiLU | PreCap B (c=1.0) | 1.00 |
| sin3_quad | 64 | GELU | CapExp (c=1.0) | 0.56 |
| sin3_quad | 64 | GELU | PreCap A (c=0.5) | 0.14 |
| sin3_quad | 64 | GELU | PreCap B (c=1.0) | 0.14 |
| sin3_quad | 64 | CapExp (c=1.0) | PreCap A (c=0.5) | 0.89 |
| sin3_quad | 64 | CapExp (c=1.0) | PreCap B (c=1.0) | 1.00 |
| sin3_quad | 64 | PreCap A (c=0.5) | PreCap B (c=1.0) | 1.00 |
| sin3_quad | 128 | SiLU | GELU | 0.09 |
| sin3_quad | 128 | SiLU | PreCap A (c=0.5) | 0.32 |
| sin3_quad | 128 | SiLU | PreCap B (c=1.0) | 1.00 |
| sin3_quad | 128 | GELU | PreCap A (c=0.5) | 0.32 |
| sin3_quad | 128 | PreCap A (c=0.5) | PreCap B (c=1.0) | 0.49 |
| sin_mix | 64 | SiLU | GELU | 1.00 |
| sin_mix | 64 | SiLU | CapExp (c=1.0) | 1.00 |
| sin_mix | 64 | GELU | CapExp (c=1.0) | 1.00 |
| sin_mix | 64 | GELU | PreCap A (c=0.5) | 0.05 |
| sin_mix | 64 | CapExp (c=1.0) | PreCap A (c=0.5) | 0.06 |
| sin_mix | 64 | CapExp (c=1.0) | PreCap B (c=1.0) | 0.07 |
| sin_mix | 64 | PreCap A (c=0.5) | PreCap B (c=1.0) | 0.72 |
| sin_mix | 128 | SiLU | GELU | 1.00 |
| sin_mix | 128 | PreCap A (c=0.5) | PreCap B (c=1.0) | 0.56 |

## Ausfaelle

| Kandidat | Tiefe | Ausfaelle | davon Schritt<=1 | Median Ausfall-Schritt |
|---|---|---|---|---|
| CapExp (c=1.0) | 64 | 11 | 3 | 740 |
| CapExp (c=1.0) | 128 | 30 | 30 | 1 |

## Dateien

summary.csv, depth_change.csv, pairwise_tests.csv, failures.csv, capexp_first_step.csv, 01-06 PNG