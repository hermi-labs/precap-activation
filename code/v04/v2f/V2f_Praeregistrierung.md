# V2f – Präregistrierung: Summieren sich die 4-Bit-Fehler bei ctanh gleichgerichteter als bei BPreCap? (Bestätigung an neuen Seeds)

Stand: 20.11.2026, 06:50. Die Präregistrierung entstand vor jedem Messergebnis an den hier gemessenen Netzen (Seeds 42353–42361). PreCap, BPreCap, die Hülle und das Kalibrierungsrezept sind Aarons Ideen. Das Messdesign habe ich mit Claude entworfen. Anlass ist die explorative Nachmessung zu V2e (`../v2e_kal/Notiz.md`, `../v2e/V2e_Auswertung.md`, Explorativ Punkt 4). Gestartet wird mit Aarons Freigabe vom 20.11. morgens („du kannst es gerne machen“, weg bis 21.11. 7:00). Eingefroren wird mit dem Start (`meta/frozen_sha256.txt`).

## Frage

In der V2e-Nachmessung wurde mit fester Kalibrierung an den Seeds 42350–42352 gemessen (explorativ, 3 Seeds). Dabei zeigte sich:
- ctanh rundet feiner, r̄ war bei BPreCap in 9 von 9 Fällen um etwa 8–14 % größer.
- Trotzdem summieren sich die Fehler über die Tiefe bei ctanh gleichgerichteter. Die Superposition D^Q / √Σd_i² war bei BPreCap in 9 von 9 Fällen kleiner.
- Bei mse4 (Abschneiden der Ränder) gibt ctanh die Fehler stärker weiter, ḡ^Q war bei BPreCap 0,75–0,86 mal so groß.

Gelten diese drei Befunde auch an Netzen, die bei der Entstehung der Hypothese nicht gemessen wurden?

## Art der Studie

Eine Messstudie ohne neue Trainingsvariante. Es gibt **genau eine Änderung gegenüber V2e**: andere Seeds (42353–42361, 9 je Form, bisher nie gemessen) und eine **feste Kalibrierung** statt der im Lauf gewählten.

Formen: BPreCap (aus V2a2) und ctanh (aus V2d), je Bestätigungslauf mit lr 3e-5 und wd 0. Die Läufe werden mit `v2e_run.run_keep` nachgerechnet. Dabei werden V2e, V2a2 und V2d unverändert importiert. Gewichte am frühen Stopp kommen in `gewichte/`.

**Kontrolle je Lauf:** Alle Messspalten der historischen CSV-Zeile (außer Laufzeit und Zeitstempel) müssen als Text gleich sein (`kontrolle_n_diff = 0`). Das ist dieselbe Funktion `v2e_run.compare` wie in V2e.

## Messung

Gemessen wird mit `v2e_run.measure` (eingefroren, unverändert): 1000 Testbilder (0–999), Gewichte fp32, nur Aktivierungen auf 4 Bit, asymmetrisch je Tensor. Kennzahlen kommen aus `v2e_run.summary`. Jedes Netz wird zweimal gemessen:
- **mse4** (primär),
- **max** (sekundär).

Die Kalibrierbereiche stammen aus `R.calib` auf den 1000 Kalibrierbildern mit dem Seed des Laufs, wie in V2e.

Kennzahlen je Lauf und Kalibrierung:
- **sup** = superpos_Q = D^Q / √(Σ_i (d_i^Q)²), mit i = 0…50.
- **ḡ^Q** = gbarQ, das geometrische Mittel von g_i^Q über i = 1…49.
- **r̄** = rbar, das geometrische Mittel von r_i über i = 1…49.
- dazu D^Q, err_a4 (Fehler auf den 1000 Bildern mit 4-Bit-Aktivierungen) und die übrigen Spalten von `summary`.

## Primärtests (BPreCap gegen ctanh, je Seed gepaart, Kalibrierung mse4)

Für jeden Seed s gilt q_s = Kennzahl(BPreCap) / Kennzahl(ctanh). Getestet wird mit einem exakten Wilcoxon-Vorzeichen-Rang-Test auf log q_s, einseitig in die vorab genannte Richtung, über 9 Paare. Die drei p-Werte werden nach Holm korrigiert.

| Test | Kennzahl | vorhergesagte Richtung |
|---|---|---|
| **H1 Kohärenz** | sup | BPreCap kleiner (q < 1) |
| **H2 Weitergabe** | ḡ^Q | BPreCap kleiner (q < 1) |
| **H3 Rundung** | r̄ | BPreCap größer (q > 1) |

**Entscheidungsregel je Test:**
- **„bestätigt“**, wenn p_Holm < 0,05 und mindestens 7 von 9 q_s in der vorhergesagten Richtung liegen.
- **„nicht bestätigt“** in allen anderen Fällen. Zusätzlich wird der zweiseitige exakte p-Wert berichtet (Hinweis auf die Gegenrichtung, ohne Urteil).
- **„unklar (Kontrolle)“**, wenn weniger als 9 gültige Paare vorliegen. Ein Paar ist gültig, wenn beide Läufe den Status ok haben, `kontrolle_n_diff = 0` ist und beide Messungen vorhanden sind.

**Gesamtdeutung (vorab):**
- **H1 und H3 bestätigt:** BPreCaps Vorteil gegen ctanh liegt nicht in feinerer Rundung, sondern darin, dass sich die Fehler über die Tiefe weniger gleichgerichtet summieren. Das ist die Hypothese aus V2e-Explorativ.
- **H1 nicht bestätigt:** Der Kohärenz-Befund aus 3 Seeds war Zufall oder hing an diesen Seeds. Im v0.4-Entwurf wird der Hinweis entfernt.
- **H2** wird für sich berichtet. Es geht darum, ob BPreCap das Abschneiden an den Rändern besser verträgt.

## Sekundär (Hinweise, eigene Holm-Korrektur je Block, gleiche Regel)

- **S1:** H1–H3 bei der Kalibrierung **max**.
- **S2:** D^Q und err_a4 bei mse4 und bei max, BPreCap kleiner (einseitig, Holm über 4).
- **S3:** gepoolt mit den 3 Seeds der V2e-Nachmessung (12 Paare), beschreibend: Median-q und die Zahl der Seeds in Richtung, je Kennzahl und Kalibrierung.
- **S4:** Spearman über die 9 Seeds zwischen log q(sup) und log q(err_a4) bei mse4, beschreibend.
- **Zerlegung:** log q(D^Q) = log q(√Σd²) + log q(sup), Mediane, beschreibend.

## Vorab-Wahrscheinlichkeiten

| Aussage | p |
|---|---|
| alle 18 Läufe bitgleich (`kontrolle_n_diff = 0`) | 0,9 |
| H1 bestätigt (sup bei mse4 BPreCap kleiner) | 0,55 |
| H2 bestätigt (ḡ^Q bei mse4 BPreCap kleiner) | 0,75 |
| H3 bestätigt (r̄ bei mse4 BPreCap größer) | 0,85 |
| S1: H1 auch bei max bestätigt | 0,5 |
| S2: D^Q bei mse4 BPreCap kleiner (bestätigt) | 0,75 |

Begründung: r̄ ist an den drei Seeds sehr gleichförmig (q 1,08–1,14), das ist eine Folge des engeren Bereichs von ctanh. Der Superpositionseffekt ist dort zwar in 9 von 9 Fällen gerichtet, aber klein (q bei mse4 0,90–0,975). Deshalb bleibt H1 unsicher. Die Hypothesen sind nach dem Ansehen von 3 Seeds entstanden, deshalb diese Bestätigung an 9 neuen Seeds.

## Grenzen

- 9 Seeds je Form, ein Netz (CNN mit 50 Schichten ohne Normierung), ein Datensatz (CIFAR-10), nur PTQ der Aktivierungen je Tensor, M1 Pro.
- Feste Kalibrierung: mse4 und max sind nicht immer die Kalibrierung, die im Lauf gewählt wurde. Der Lauf-Preis aus V2a2 und V2d enthält auch w8.
- Superposition ist eine Größe über Frobenius-Normen. Sie sagt nicht, in welcher Richtung im Logit-Raum sich die Fehler addieren.
- Es ist eine Mechanismus-Beschreibung, kein kausaler Eingriff.

## Ablauf und Soll

1. `launch_v2f.sh` prüft Netzteil, andere Worker und die Hashes von V2e, V2a2 und V2d. Dann folgen Freeze, Selbsttest, Auswertungstest mit künstlichen Daten, Rauchtest (2 Läufe à 20 Schritte, 250 Bilder, beide Kalibrierungen) und der Start in screen `v2fw0` mit caffeinate und dem Netzteil-Wächter `v2fwache`. Es läuft 1 Worker.
2. Reihenfolge: je Seed erst BPreCap, dann ctanh, Seeds aufsteigend. **Soll: 36 Zeilen** (18 Läufe × 2 Kalibrierungen).
3. Erwartete Dauer etwa 11–12 h (BPreCap etwa 51 min, ctanh etwa 27 min je Seed, Messung eingeschlossen).
4. Auswertung mit `v2f_analyze.py <V2f-Ordner absolut>` (scipy-venv), Hash-Prüfung vorher und nachher, Bericht `V2f_Auswertung.md`.
