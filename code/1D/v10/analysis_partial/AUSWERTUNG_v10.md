# v10 Auswertung (VORLAEUFIG)

Stand: 2026-10-19 21:04. Runs: **6227 von 6360**, Duplikate: 0, fehlende Konfigurationen: 133, Ausfaelle (bad): 0.

## Schritt 1 - Integritaet
| shard | rows | bad | mean_sec |
|---|---|---|---|
| 0 | 880 | 0 | 57.2 |
| 1 | 897 | 0 | 56.1 |
| 2 | 894 | 0 | 56.3 |
| 3 | 896 | 0 | 56.2 |
| 4 | 887 | 0 | 56.7 |
| 5 | 883 | 0 | 57 |
| 6 | 890 | 0 | 56.3 |

## Schritt 2 - Typische Qualitaet (Median der Konfig-Mediane)
| depth | arch | act | abs_wave | sin3_quad | sin_mix |
|---|---|---|---|---|---|
| 4 | plain | capexp | 0.00105 | 2.8e-05 | 0.00128 |
| 4 | plain | gelu | 0.000111 | 1.11e-05 | 0.000543 |
| 4 | plain | precap | 0.000221 | 0.000169 | 0.000878 |
| 4 | plain | relu | 0.000762 | 0.000562 | 0.000621 |
| 4 | plain | silu | 0.000188 | 9.38e-06 | 0.000534 |
| 4 | res_inv | capexp | 0.00151 | 0.000271 | 0.00157 |
| 4 | res_inv | gelu | 0.0103 | 0.00108 | 0.000872 |
| 4 | res_inv | precap | 0.000195 | 0.000126 | 0.000849 |
| 4 | res_inv | relu | 0.0264 | 0.00157 | 0.00199 |
| 4 | res_inv | silu | 0.00288 | 0.000439 | 0.00122 |
| 8 | plain | capexp | 0.000308 | 4.81e-05 | 0.000499 |
| 8 | plain | gelu | 0.000137 | 6.21e-05 | 0.000433 |
| 8 | plain | precap | 0.000658 | 0.000754 | 0.000702 |
| 8 | plain | relu | 0.00118 | 0.0015 | 0.000647 |
| 8 | plain | silu | 0.000164 | 3.36e-05 | 0.000359 |
| 8 | res_inv | capexp | 0.000231 | 0.000106 | 0.00115 |
| 8 | res_inv | gelu | 0.00526 | 0.00177 | 0.00111 |
| 8 | res_inv | precap | 0.000111 | 0.00014 | 0.000615 |
| 8 | res_inv | relu | 0.0184 | 0.00131 | 0.00229 |
| 8 | res_inv | silu | 0.000649 | 0.000679 | 0.00136 |
| 32 | res_inv | gelu | 0.000606 | 0.000549 | 0.00127 |
| 32 | res_inv | precap | 7.21e-05 | 6.13e-05 | 0.000751 |
| 32 | res_inv | silu | 0.000354 | 0.000108 | 0.00153 |
| 64 | res_inv | gelu | nan | 0.000302 | 0.00124 |
| 64 | res_inv | precap | 7.54e-05 | 6.19e-05 | 0.000626 |
| 64 | res_inv | silu | nan | 4.36e-05 | 0.0014 |

## Beste Konfiguration je Aufgabe/Tiefe
| target | depth | arch | act | c | a | gamma | soft_s | init_k | lr | med | q25 | q75 | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abs_wave | 4 | plain | gelu | 0 | 1 | 0 | 1 | 1 | 0.003 | 7.02e-05 | 5.96e-05 | 0.000112 | 0 |
| abs_wave | 8 | plain | gelu | 0 | 1 | 0 | 1 | 1 | 0.001 | 3.4e-05 | 2.55e-05 | 5.13e-05 | 0 |
| abs_wave | 32 | res_inv | precap | 1.25 | 1.25 | 0.01 | 1.5 | 0.5 | 0.001 | 4.07e-05 | 3.25e-05 | 7.43e-05 | 0 |
| abs_wave | 64 | res_inv | precap | 1.25 | 1.25 | 0 | 1.5 | 0.5 | 0.001 | 2.76e-05 | 2.5e-05 | 3.61e-05 | 0 |
| sin3_quad | 4 | plain | silu | 0 | 1 | 0 | 1 | 1 | 0.003 | 5.79e-06 | 4.8e-06 | 7.21e-06 | 0 |
| sin3_quad | 8 | plain | silu | 0 | 1 | 0 | 1 | 1 | 0.001 | 5.22e-06 | 4.02e-06 | 5.64e-06 | 0 |
| sin3_quad | 32 | res_inv | precap | 1.25 | 1 | 0 | 2 | 0.5 | 0.001 | 6.57e-06 | 5.47e-06 | 7.24e-06 | 0 |
| sin3_quad | 64 | res_inv | precap | 1.25 | 1 | 0.01 | 1.5 | 0.5 | 0.001 | 4.61e-06 | 3.38e-06 | 7.98e-06 | 0 |
| sin_mix | 4 | plain | silu | 0 | 1 | 0 | 1 | 0.5 | 0.001 | 0.000415 | 0.000372 | 0.000498 | 0 |
| sin_mix | 8 | plain | silu | 0 | 1 | 0 | 1 | 1 | 0.003 | 0.000118 | 9.76e-05 | 0.000232 | 0 |
| sin_mix | 32 | res_inv | precap | 1.25 | 1.5 | 0.02 | 2 | 0.5 | 0.001 | 0.00048 | 0.00045 | 0.000527 | 0 |
| sin_mix | 64 | res_inv | precap | 1 | 1.5 | 0.02 | 1.5 | 0.5 | 0.001 | 0.000467 | 0.000426 | 0.00055 | 0 |

## Schritt 3 - Stabilitaet
| act | arch | depth | mean | sum | count |
|---|---|---|---|---|---|
| capexp | plain | 4 | 0 | 0 | 120 |
| capexp | plain | 8 | 0 | 0 | 120 |
| capexp | res_inv | 4 | 0 | 0 | 120 |
| capexp | res_inv | 8 | 0 | 0 | 120 |
| gelu | plain | 4 | 0 | 0 | 120 |
| gelu | plain | 8 | 0 | 0 | 120 |
| gelu | res_inv | 4 | 0 | 0 | 120 |
| gelu | res_inv | 8 | 0 | 0 | 120 |
| gelu | res_inv | 32 | 0 | 0 | 60 |
| gelu | res_inv | 64 | 0 | 0 | 40 |
| precap | plain | 4 | 0 | 0 | 240 |
| precap | plain | 8 | 0 | 0 | 240 |
| precap | res_inv | 4 | 0 | 0 | 240 |
| precap | res_inv | 8 | 0 | 0 | 240 |
| precap | res_inv | 32 | 0 | 0 | 1620 |
| precap | res_inv | 64 | 0 | 0 | 1527 |
| relu | plain | 4 | 0 | 0 | 120 |
| relu | plain | 8 | 0 | 0 | 120 |
| relu | res_inv | 4 | 0 | 0 | 120 |
| relu | res_inv | 8 | 0 | 0 | 120 |
| silu | plain | 4 | 0 | 0 | 120 |
| silu | plain | 8 | 0 | 0 | 120 |
| silu | res_inv | 4 | 0 | 0 | 120 |
| silu | res_inv | 8 | 0 | 0 | 120 |
| silu | res_inv | 32 | 0 | 0 | 60 |
| silu | res_inv | 64 | 0 | 0 | 40 |

| act | arch | depth | grad_final_med | absmax_med | absmax_p95 | cap_frac_med | grad_p99_alle_steps | gen_gap_med |
|---|---|---|---|---|---|---|---|---|
| capexp | plain | 4 | 0.0118 | 31.5 | 76 | 0.0423 | 4.94 | 2.23e-05 |
| capexp | plain | 8 | 0.0963 | 30.1 | 101 | 0.0244 | 8.51 | 6.8e-05 |
| capexp | res_inv | 4 | 0.0357 | 33.3 | 62.1 | 0.0947 | 37.1 | 0.000336 |
| capexp | res_inv | 8 | 0.0544 | 37.3 | 63.9 | 0.0763 | 69.7 | 0.000112 |
| gelu | plain | 4 | 0.00298 | 5.2 | 8.97 | 0 | 2.63 | 3.04e-05 |
| gelu | plain | 8 | 0.0226 | 5.2 | 8.35 | 0 | 1.87 | 6.66e-05 |
| gelu | res_inv | 4 | 0.0404 | 10.7 | 18.9 | 0 | 82.9 | 0.000798 |
| gelu | res_inv | 8 | 0.116 | 12.3 | 21.6 | 0 | 113 | 0.000904 |
| gelu | res_inv | 32 | 0.157 | 18.6 | 34.1 | 0 | 1.11e+03 | 0.000473 |
| gelu | res_inv | 64 | 0.306 | 27.5 | 41.8 | 0 | 6.12e+03 | 0.000772 |
| precap | plain | 4 | 0.019 | 2.67 | 3.08 | 0.174 | 49.2 | 0.000206 |
| precap | plain | 8 | 0.138 | 2.63 | 3.08 | 0.188 | 84.2 | 0.00053 |
| precap | res_inv | 4 | 0.000758 | 6.54 | 8.01 | 0.231 | 77.5 | 0.000208 |
| precap | res_inv | 8 | 0.00233 | 7.6 | 9.61 | 0.235 | 82.3 | 0.000179 |
| precap | res_inv | 32 | 0.0203 | 11.8 | 16.2 | 0.097 | 128 | 6.97e-05 |
| precap | res_inv | 64 | 0.0543 | 13.2 | 18.9 | 0.0898 | 264 | 7.78e-05 |
| relu | plain | 4 | 0.096 | 4.34 | 5.79 | 0 | 3.73 | 0.000572 |
| relu | plain | 8 | 0.14 | 3.76 | 5.8 | 0 | 2.91 | 0.000844 |
| relu | res_inv | 4 | 0.0979 | 8.57 | 11.7 | 0 | 93.1 | 0.00132 |
| relu | res_inv | 8 | 0.286 | 8.41 | 13.7 | 0 | 136 | 0.00101 |
| silu | plain | 4 | 0.00148 | 6.89 | 16.1 | 0 | 2.14 | 1.9e-05 |
| silu | plain | 8 | 0.0152 | 6.56 | 15 | 0 | 1.52 | 4.91e-05 |
| silu | res_inv | 4 | 0.027 | 14.1 | 26.1 | 0 | 70 | 0.000875 |
| silu | res_inv | 8 | 0.0654 | 14.5 | 30.1 | 0 | 91.7 | 0.000462 |
| silu | res_inv | 32 | 0.0299 | 19.7 | 35.7 | 0 | 774 | 0.00011 |
| silu | res_inv | 64 | 0.0327 | 26.1 | 39.6 | 0 | 4.13e+03 | 0.000799 |

## Schritt 4 (vorlaeufig) - beste PreCap-Konfig vs. beste Konfig der Gegner, Seed-gepaart
| target | depth | arch | gegner | precap_med | gegner_med | faktor | seed_siege | n_paare | p_wilcoxon |
|---|---|---|---|---|---|---|---|---|---|
| abs_wave | 4 | plain | gelu | 0.000117 | 7.02e-05 | 0.599 | 0.3 | 10 | 0.105 |
| abs_wave | 4 | plain | silu | 0.000117 | 9.87e-05 | 0.842 | 0.5 | 10 | 1 |
| abs_wave | 4 | res_inv | gelu | 0.00011 | 0.00106 | 9.63 | 1 | 10 | 0.00195 |
| abs_wave | 4 | res_inv | silu | 0.00011 | 0.000614 | 5.57 | 1 | 10 | 0.00195 |
| sin3_quad | 4 | plain | gelu | 2.8e-05 | 8.45e-06 | 0.302 | 0.2 | 10 | 0.0195 |
| sin3_quad | 4 | plain | silu | 2.8e-05 | 5.79e-06 | 0.207 | 0 | 10 | 0.00195 |
| sin3_quad | 4 | res_inv | gelu | 7.26e-05 | 0.000302 | 4.16 | 1 | 10 | 0.00195 |
| sin3_quad | 4 | res_inv | silu | 7.26e-05 | 0.000212 | 2.91 | 0.9 | 10 | 0.00391 |
| sin_mix | 4 | plain | gelu | 0.000687 | 0.000489 | 0.711 | 0.2 | 10 | 0.0645 |
| sin_mix | 4 | plain | silu | 0.000687 | 0.000415 | 0.604 | 0.1 | 10 | 0.0195 |
| sin_mix | 4 | res_inv | gelu | 0.000695 | 0.00082 | 1.18 | 0.6 | 10 | 0.432 |
| sin_mix | 4 | res_inv | silu | 0.000695 | 0.00107 | 1.53 | 0.9 | 10 | 0.00586 |
| abs_wave | 8 | plain | gelu | 0.000204 | 3.4e-05 | 0.166 | 0.1 | 10 | 0.00586 |
| abs_wave | 8 | plain | silu | 0.000204 | 0.000114 | 0.56 | 0.1 | 10 | 0.00977 |
| abs_wave | 8 | res_inv | gelu | 6.7e-05 | 0.00146 | 21.8 | 0.9 | 10 | 0.00391 |
| abs_wave | 8 | res_inv | silu | 6.7e-05 | 0.000559 | 8.34 | 0.9 | 10 | 0.0273 |
| sin3_quad | 8 | plain | gelu | 0.000155 | 1.85e-05 | 0.12 | 0.3 | 10 | 0.0488 |
| sin3_quad | 8 | plain | silu | 0.000155 | 5.22e-06 | 0.0337 | 0.1 | 10 | 0.00586 |
| sin3_quad | 8 | res_inv | gelu | 6.76e-05 | 0.00133 | 19.6 | 0.9 | 10 | 0.00391 |
| sin3_quad | 8 | res_inv | silu | 6.76e-05 | 0.000189 | 2.79 | 0.9 | 10 | 0.0645 |
| sin_mix | 8 | plain | gelu | 0.000446 | 0.000177 | 0.398 | 0.1 | 10 | 0.00391 |
| sin_mix | 8 | plain | silu | 0.000446 | 0.000118 | 0.265 | 0 | 10 | 0.00195 |
| sin_mix | 8 | res_inv | gelu | 0.00052 | 0.00106 | 2.03 | 1 | 10 | 0.00195 |
| sin_mix | 8 | res_inv | silu | 0.00052 | 0.00117 | 2.26 | 1 | 10 | 0.00195 |
| abs_wave | 32 | res_inv | gelu | 4.07e-05 | 0.000393 | 9.67 | 0.9 | 10 | 0.00586 |
| abs_wave | 32 | res_inv | silu | 4.07e-05 | 0.000291 | 7.16 | 0.9 | 10 | 0.0195 |
| sin3_quad | 32 | res_inv | gelu | 6.57e-06 | 0.000363 | 55.3 | 1 | 10 | 0.00195 |
| sin3_quad | 32 | res_inv | silu | 6.57e-06 | 8.71e-05 | 13.3 | 0.9 | 10 | 0.00391 |
| sin_mix | 32 | res_inv | gelu | 0.00048 | 0.00121 | 2.52 | 1 | 10 | 0.00195 |
| sin_mix | 32 | res_inv | silu | 0.00048 | 0.00135 | 2.81 | 1 | 10 | 0.00195 |
| sin3_quad | 64 | res_inv | gelu | 4.61e-06 | 0.000141 | 30.6 | 1 | 10 | 0.00195 |
| sin3_quad | 64 | res_inv | silu | 4.61e-06 | 4.01e-05 | 8.68 | 0.9 | 10 | 0.00391 |
| sin_mix | 64 | res_inv | gelu | 0.000467 | 0.00115 | 2.47 | 1 | 10 | 0.00195 |
| sin_mix | 64 | res_inv | silu | 0.000467 | 0.00101 | 2.17 | 1 | 10 | 0.00195 |

Hinweis: 'beste Konfig' wird anhand derselben Seeds gewaehlt, p-Werte sind daher optimistisch. PreCap hat in v10 deutlich mehr Hyperparameter-Varianten als SiLU/GELU - gepoolte Medians ueber alle Zeilen sind verzerrt, deshalb hier Konfig-Mediane.
