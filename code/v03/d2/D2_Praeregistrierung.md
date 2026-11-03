# D2 – Präregistrierung: Warum bricht BPreCap bei einer LR-Spitze von 2·10⁻⁴ ein? (beschreibend, privat)

Stand: 02.11.2026, nachts, vor dem Start. Die Freeze-Zeit steht in `meta/frozen_at.txt`. Entwurf und Lauf mit Claude, in Aarons Abwesenheit nach den Leitlinien für das Alleinarbeiten. Grundlage: B10-Tuning, Teil von v0.4/V1 im Fahrplan.

## Befund, der erklärt werden soll

Im B10-Tuning (CIFAR-10, d100, TAT-Rezept, AdamW, Kosinus, 2 Seeds) liegt der Val-Fehler am Ende bei:

| Form | LR 1e-4, WD 0,3 | LR 2e-4, WD 0,3 | LR 2e-4, WD 3,0 |
|---|---|---|---|
| BPreCap (kL2B_tr) | 45,8 / 45,3 % | 84,0 / 81,7 % | 56,2 / 58,6 % |
| LB_tr (linearer Kern, gleiche Hülle) | 47,7 / 48,2 % | 86,4 / 81,7 % | 63,6 / 73,8 % |
| TAT (2e-4, WD 0) | | 45,7 / 46,5 % | |

Auch LB bricht ein, bei gleicher Hülle und anderem Kern. Starker WD rettet beide teilweise.

## Frage und Hypothesen (vorab)

**Frage:** Was passiert im Netz, wenn BPreCap bei 2e-4 einbricht?

| Nr. | Hypothese | Erwartetes Bild | Vorab-Wahrscheinlichkeit |
|---|---|---|---|
| H-a | **Sättigung der Hülle:** Viele Werte laufen tief in die Sättigung (\|y\| > 0,99·M). Die Ableitung der Hülle geht gegen 0, Gradienten verschwinden, das Netz bleibt stecken | deep_med steigt bei 2e-4 deutlich über den Wert bei 1e-4. Der Gradient der vorderen Schichten fällt (glog_first_last sinkt) | 0,55 |
| H-b | **Gewichte wachsen** (Adam-Schritte zu groß): Die Vor-Aktivierungen werden groß. Das ist die Ursache von H-a | wnorm_med und prerms_med steigen bei 2e-4 früh und stark. WD 3,0 bremst das | 0,5 (nicht exklusiv zu H-a) |
| H-c | **Früher Gradientenausbruch** (Explosion statt Stillstand) | Spitzen in glog oder im Verlust in den ersten Hundert Schritten, nicht-endliche Werte | 0,2 |
| H-d | **Kein Hüllen-Effekt:** Der harte Clip (bh2) bricht genauso ein | bh2 ähnlich bp2 | 0,5 |

## Läufe

| Name | Form | LR-Spitze | WD | Quelle der Vergleichszeile |
|---|---|---|---|---|
| bp1 | kL2B_tr | 1e-4 | 0,3 | B10-Tuning |
| bp2 | kL2B_tr | 2e-4 | 0,3 | B10-Tuning |
| bp2w3 | kL2B_tr | 2e-4 | 3,0 | B10-Tuning |
| lb2 | LB_tr | 2e-4 | 0,3 | B10-Tuning |
| bh2 | kL2Bh_tr (harter Clip bei M) | 2e-4 | 0,3 | B13-Tuning |
| ta2 | tat | 2e-4 | 0,0 | B10-Tuning |

- **Seeds:** 42000 und 42001 (Tuning-Seeds), also 12 Läufe.
- **Training:** exakt wie im Tuning. Das Skript lädt `b10_run.py` bzw. `b13_run.py` unverändert.
- **Gültigkeit:** Ein Lauf gilt, wenn der Val-Fehler am Ende **bitgleich** zur Tuning-Zeile ist. Die Messungen ändern das Training nicht, sie rechnen nur vorwärts und Gradienten ohne Aktualisierung.

## Messung

Alle 86 Schritte auf einem festen Probe-Batch (die ersten 512 Trainingsbilder):
- Verlust;
- **Sättigung:** Anteil von \|y\| > 0,8·M (Knie) bzw. > 0,99·M (tief), M = 3, je Schicht. Ausgewertet werden Median, Maximum und 6 feste Schichten. Für TAT dienen dieselben Schwellen nur zur Einordnung;
- RMS der Vor-Aktivierung (Median, Maximum);
- Gradientennormen der Gewichte je Schicht: log10 erste/letzte Schicht, Median, 6 feste Schichten;
- Frobenius-Norm der Gewichte (Median, Maximum).

## Auswertung (beschreibend, vorab festgelegt)

- **H-a gilt als gestützt,** wenn in beiden Seeds gilt: deep_med von bp2 ist am Ende mindestens 5-mal so groß wie bei bp1 und über 5 %, und der Median des Gradienten (glog_med) liegt am Ende mindestens 1 Dekade unter bp1.
- **H-b gilt als gestützt,** wenn wnorm_med von bp2 in beiden Seeds bis Schritt 1720 mindestens 1,5-mal so groß ist wie bei bp1 und bp2w3 dazwischen liegt.
- **H-c gilt als gestützt,** wenn es nicht-endliche Werte gibt oder glog_med in den ersten 860 Schritten um mehr als 2 Dekaden springt.
- **H-d:** Der Val-Fehler von bh2 liegt im Bereich von bp2, beschreibend.
- **Daraus folgt die Hypothese für die V1-Studie**, vorab grob:
  - H-a/H-b gestützt: eine Änderung an der LR (Anlauf/Warmup, kleinere LR für die vorderen Schichten) oder am WD prüfen;
  - H-c: Gradient-Clipping prüfen.

## Grenzen

- 2 Seeds, nur CIFAR-10 d100, nur das TAT-Rezept.
- Beschreibend, ohne Test. Ergebnisse sind Hinweise für die V1-Studie, keine Belege.
