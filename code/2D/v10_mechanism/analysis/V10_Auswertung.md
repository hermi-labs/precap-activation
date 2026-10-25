# V10 Auswertung (160/160 Runs, 1120/1120 Checkpoints, keine NaN/Inf, Hashes OK)

## Leistung (Step 1200, d = log10(MSE_s225 / MSE_base), negativ = s225 besser)
| Aufgabe | Tiefe | Median d | Median-Verhaeltnis | Seeds s225 besser | p (einseitig) |
|---|---|---|---|---|---|
| geo_a025 | 16 | -0.281 | 0.52 | 18/20 | <0.001 |
| geo_a025 | 32 | -0.573 | 0.27 | 17/20 | <0.001 |
| geo_a075 | 16 | +0.038 | 1.09 | 8/20 | 0.90 |
| geo_a075 | 32 | +0.049 | 1.12 | 9/20 | 0.81 |

Die V9-Beobachtung wurde mit neuen Seeds (5000-5019) und beiden Tiefen repliziert: s225 hilft klar bei a=0.25, nicht bei a=0.75 (tendenziell leicht schlechter).

## Mechanismus (Median s225 minus base)
- Kappungsanteil: bei a025 praktisch gleich zu base (Differenz ca. 0.000 bis 0.003), bei a075 dauerhaft hoeher (+0.004 bis +0.005). 
- Pre-Activation p99: bei a025 durchgehend niedriger (bis -0.21 bei Tiefe 32), bei a075 Tiefe 32 hoeher (ca. +0.07).
- hmax_peak: bei a025 ab Step 200 niedriger (ca. -0.06 bis -0.11), bei a075 hoeher (+0.13 bis +0.18).
- Gradienten-Norm: Differenzen klein und vorzeichenwechselnd, kein klares Muster ausser a025 Tiefe 32 (spaet negativ, -0.19).
- Post-Activation p99: in beiden Aufgaben spaeter hoeher, nicht diskriminierend.
- Seed-Korrelation zwischen Leistungsgewinn d und Kappungsdifferenz: a025 d16 -0.57, a025 d32 -0.20, a075 +0.17 / -0.02.

## Einordnung
Kandidat (explorativ): s225 reduziert bei a025 die extremen Pre-Activation-Werte und den Aktivierungspeak, bei a075 erhoeht es sie und die Kappung. Die Seed-Korrelationen sind nur bei a025 Tiefe 16 nennenswert; kein bewiesener Kausalmechanismus. Mediane ohne Mehrfachtest-Korrektur.

## Entscheidung
- precap_base bleibt Standard; s225 ist aufgabenspezifisch (replizierter Vorteil bei a025, kein Vorteil bei a075).
- Naechster Schritt V11: kleine Bestaetigung mit neuen Seeds und einer unabhaengigen Aufgabenfamilie sowie praeregistrierter Hypothese zu Pre-Activation-p99 / hmax_peak.
