# V10 Auswertungs- und Entscheidungsplan

## Ziel
V10 ist eine Mechanismusstudie. Sie soll erklaeren, warum `precap_s225` bei V9 auf `geo_a025` stark half, bei `geo_a075` aber nicht verlaesslich half. Sie ist nicht dazu gedacht, einen neuen globalen Standardparameter zu kueren.

## Datenintegritaet vor jeder Aussage
1. 160/160 Runs und 1120/1120 Checkpoint-Zeilen.
2. Eindeutiger Schluessel `(run_id, step)` fuer jede Zeile.
3. Keine NaN/Inf in Loss-, Gradienten-, Kappungs- oder Quantilspalten.
4. Hashes des eingefrorenen Runners, Launchers, Statusskripts und der Praeregistrierung stimmen.
5. Ergebnisse aus `_invalid_smoketest/` werden nie eingelesen.

## Primaere Leistungs-Replikation
Fuer jede Kombination aus Aufgabe und Tiefe wird pro Seed berechnet:

`d = log10(test_mse_s225 / test_mse_base)`

Negatives d bedeutet Vorteil fuer s225. Berichtet werden Median d, Anzahl Seeds mit d<0 und einseitiger gepaarter Wilcoxon-Test gegen 0. Diese Replikation dient nur als Anker: V10 macht keine neue globale Leistungsbehauptung.

## Primaere Mechanismusprofile
Fuer jeden Checkpoint werden pro Aufgabe, Tiefe und Aktivierung robuste Seed-Mediane samt IQR fuer diese Groessen verglichen:
- globale Gradienten-Norm,
- Kappungsanteil,
- Pre-Activation-Quantile p01/p50/p99,
- Post-Activation-Quantile p01/p50/p99,
- hmax_peak.

Die entscheidende Darstellung ist jeweils die Differenz `s225 - base` als Zeitprofil, separat fuer `geo_a025` und `geo_a075` sowie Tiefe 16 und 32.

## Mechanismus-Kriterien
Ein brauchbarer Kandidatenmechanismus muss:
1. in der a025-Erfolgszelle mit dem Leistungsgewinn zusammen auftreten,
2. in a075 fehlen, umgekehrt sein oder deutlich schwaecher ausfallen,
3. in mindestens beiden Tiefen die gleiche Richtung haben oder einen klar dokumentierten Tiefenunterschied zeigen,
4. nicht nur an einem einzelnen Checkpoint oder wenigen Seeds beruhen.

Beispiele: gleichmaessigerer Gradientenerhalt, andere Kappungsrate oder andere tiefe Aktivierungsstreuung. Ein bloßer Unterschied ohne Replikation ueber Tiefe oder Seeds gilt nur explorativ.

## Was nicht behauptet wird
- Kein Schluss, dass s225 global besser ist.
- Kein Schluss aus einer einzelnen Schicht, einem einzelnen Seed oder einem einzelnen Checkpoint.
- Kein automatisches neues Parametersweeping, falls sich kein klarer Mechanismus zeigt.

## Nach dem Lauf
1. Integritaets- und Hashpruefung.
2. Ergebnisse zusammenfuehren.
3. Eine feste Analyse erzeugen: Leistungszellen, Zeitprofile, Seed-Streuung, kurze Endauswertung.
4. Entscheidung:
   - Klarer Mechanismus: V11 als kleine, bestaetigende Studie auf einem neuen Aufgaben-/Seed-Satz.
   - Kein klarer Mechanismus: base bleibt Standard; s225 bleibt dokumentierte Spezialvariante.
