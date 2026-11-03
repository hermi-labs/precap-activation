# D2 – Auswertung: Warum bricht BPreCap bei einer LR-Spitze von 2·10⁻⁴ ein? (beschreibend)

Stand: 02.11.2026, nachts. Auswertung nach `D2_Praeregistrierung.md` (eingefroren 21:37). Lauf und Auswertung mit Claude, in Aarons Abwesenheit.

## Kurz

- **Die Hülle ist nicht die Ursache.** LB bricht genauso ein (Val-Fehler 86 und 82 %), obwohl seine Werte die Hülle **nie** erreichen: Der Anteil im Knie liegt durchgehend bei 0,00. Der harte Clip (bh2) bricht ebenfalls ein (83 und 84 %).
- **Was passiert:** Ab etwa Schritt 430 (LB) bzw. 860–3440 (BPreCap) **verschwinden die Gradienten**.
  - Der Median über die Schichten fällt auf 10⁻³·⁴ bis 10⁻⁴·⁵. Bei LR 1e-4 liegt er stabil bei 10⁻⁰·⁵ bis 10⁻⁰·⁷.
  - Die erste Schicht bekommt 10⁴- bis 10⁷-mal weniger Gradient als die letzte. Bei 1e-4 ist es umgekehrt: 10⁰·⁴ bis 10⁰·⁹ mehr.
  - Der Verlust steigt danach zurück auf etwa 2,2, nahe an ln 10 = 2,30 (Raten).
- **Kein Ausbruch, kein Gewichtswachstum:** Es gibt keine nicht-endlichen Werte und keine Spitzen nach oben. Die Gewichtsnormen schrumpfen bei 2e-4 sogar etwas stärker als bei 1e-4 (14,5 gegen 15,3 bei Schritt 1720).
- **Starker WD (3,0) rettet teilweise,** indem er die Gewichte stark verkleinert: Frobenius-Norm 16 → 2,8, RMS der Vor-Aktivierung 0,6 → 0,2. Das Netz läuft dann fast linear und klein, die Gradienten bleiben gesund.
- **TAT trainiert bei 2e-4 normal,** mit Gradienten-Median 10⁻⁰·⁶ bis 10⁻¹·⁰.

## Integrität

| Prüfung | Ergebnis |
|---|---|
| Läufe | 12 von 12 ok |
| Bitgleich zur Tuning-Zeile (Val-Fehler am Ende) | **12 von 12** (B10- bzw. B13-Tuning) |
| Hash-Prüfung | 4 von 4 OK (`d2_diag.py`, Präregistrierung, `b10_run.py`, `b13_run.py`) |
| Laufzeit | 21:37–22:06, 2 Worker, M1 Pro am Netzteil |

## Verläufe (Seed 42000; Seed 42001 gleiches Bild, siehe `results/`)

| Lauf | Val-Fehler | Gradient-Median log10 (Schritt 430 / 1720 / 6880) | erste/letzte Schicht log10 (1720) | tiefe Sättigung Median (1720) | Gewichtsnorm Median (1720) |
|---|---|---|---|---|---|
| bp1 (1e-4, WD 0,3) | 45,8 % | −0,5 / −0,6 / −0,7 | +0,4 | 0,7 % | 15,3 |
| bp2 (2e-4, WD 0,3) | 84,0 % | −0,5 / **−3,4** / −3,3 | **−4,1** | 4,1 % | 14,5 |
| bp2w3 (2e-4, WD 3,0) | 56,2 % | −0,7 / −0,6 / −1,0 | +0,5 | 0,0 % | 6,3 |
| lb2 (LB, 2e-4, WD 0,3) | 86,4 % | **−3,5** / −4,4 / −3,0 | **−7,2** | **0,0 %** | 14,5 |
| bh2 (harter Clip, 2e-4) | 83,2 % | −0,6 / −1,0 / **−4,3** | 0,0 (3440: −5,1) | 6,6 % | 14,5 |
| ta2 (TAT, 2e-4, WD 0) | 45,7 % | −0,6 / −0,9 / −1,0 | +0,6 | (7,1 %, ohne Grenze nur Einordnung) | 15,8 |

## Hypothesen nach den Vorab-Kriterien

| Nr. | Kriterium | Ergebnis |
|---|---|---|
| H-a Sättigung der Hülle | deep_med am Ende mindestens 5-mal bp1 **und** über 5 %, **und** Gradient ≥ 1 Dekade unter bp1, in beiden Seeds | **nicht gestützt.** Der Gradient-Teil ist erfüllt (−3,3/−3,9 gegen −0,7). Die Sättigung am Ende liegt aber nur bei 0,2 % bzw. 3,2 %, und LB zeigt den Einbruch ganz **ohne** Sättigung |
| H-b Gewichtswachstum | wnorm bp2 bis 1720 ≥ 1,5 × bp1 | **nicht gestützt.** 14,5 gegen 15,3, die Gewichte wachsen nicht |
| H-c Gradientenausbruch | nicht-endliche Werte oder Sprung > 2 Dekaden in den ersten 860 Schritten | **nicht gestützt.** Es gibt keinen Ausbruch nach oben. Der Gradient fällt, er explodiert nicht |
| H-d unabhängig von der Hüllenform | bh2 im Bereich von bp2 | **gestützt.** 83,2/84,5 % gegen 84,0/81,7 % |

**Keine der vorab genannten Ursachen trägt.** Die vorab festgelegte Zuordnung zur V1-Hypothese (H-a/H-b → LR-Änderung, H-c → Clipping) greift damit nicht. Das wird offen so festgehalten.

## Neue Hypothese (nach dem Befund, nicht vorab)

**H-e, Verlust der Isometrie:** Große Adam-Schritte zerstören früh im Training die fast orthogonale Struktur der Gewichtsmatrizen. Das Produkt der Jacobi-Matrizen über 100 Schichten zieht sich dann zusammen, und die vorderen Schichten bekommen keinen Gradienten mehr.
- **Dafür spricht:**
  - Der Einbruch tritt in Netzen auf, die fast linear sind (LB: β·x, die Hülle bleibt ungenutzt).
  - Er ist unabhängig von der Form der Grenze.
  - Kleine Gewichte (WD 3,0) verhindern ihn teilweise.
- **Warum TAT überlebt,** ist offen. Möglich ist, dass die stückweise lineare Form die Signal-Geometrie anders hält.
- **Explorativ geprüft in D2b** (unten): Spektrum der Gewichte und ein linearer LR-Anlauf (Warmup) als Gegenmittel.

## Grenzen

- 2 Seeds, nur CIFAR-10 d100, TAT-Rezept, Messung alle 86 Schritte. Kurze Spitzen dazwischen sind nicht erfasst.
- Die Sättigungsschwellen gelten für M = 3. Für TAT sind sie nur zur Einordnung.

## D2b – explorativ (nicht präregistriert, ein Seed)

Skript `d2b_explor.py` (wie `d2_diag.py`, dazu Singulärwerte der Gewichte in Schicht 1, 25, 50, 75, 100 und optional ein linearer LR-Anlauf über 344 Schritte = 5 %, danach Kosinus). Seed 42000, 22:07–22:16. Die drei Läufe ohne Anlauf sind bitgleich zu D2.

| Lauf | ohne Anlauf | **mit 5 % Anlauf** |
|---|---|---|
| BPreCap, 2e-4, WD 0,3 | 84,0 % | **45,9 %** |
| LB, 2e-4, WD 0,3 | 86,4 % | **48,7 %** |
| zum Vergleich BPreCap, 1e-4, WD 0,3 (B10-Wahl) | 45,8 % | – |

- **Ein linearer Anlauf über 5 % der Schritte verhindert den Einbruch vollständig,** in diesem einen Seed. BPreCap erreicht bei 2e-4 denselben Val-Fehler wie die B10-Wahl mit 1e-4. Die Gradienten bleiben gesund (Median 10⁻⁰·⁴ bis 10⁻¹·⁰, erste/letzte Schicht +0,2 bis +0,9).
- **Spektrum:** Ohne Anlauf wächst in den ersten Schritten vor allem die **Eingangsschicht** (3072 → 256):

  | Lauf | größter Singulärwert, Schritt 172 | Schritt 1720 |
  |---|---|---|
  | LB ohne Anlauf | 1,85 | 3,93 |
  | BPreCap ohne Anlauf | 1,27 | 3,92 |
  | mit Anlauf | 1,03 | 1,6–1,7 |
  | bp1 | 1,08 | 1,38 |

  Die Schichten in der Mitte bleiben in allen Läufen nahe isometrisch (größter Singulärwert 1,0–1,2, Kondition im Median 1,2–1,4, bei LB 2,4). **H-e im strengen Sinn,** also Isometrieverlust überall, trifft damit nicht zu. Auffällig ist der frühe Sprung der Eingangsschicht, und der Gradient bricht vor allem vorne ein.
- **Lesart (vorläufig):** Die ersten, ungedämpften Adam-Schritte bei voller LR stoßen das Netz aus dem kritischen Bereich, vor allem über die Eingangsschicht. Danach verschwinden die Gradienten nach vorne. Ein Anlauf vermeidet diese frühen Stöße. Das ist in der Praxis gut bekannt (Warmup bei Adam, z. B. Liu et al. 2020, RAdam). Für BPreCap ist es hier zum ersten Mal gemessen.
- **Vorsicht:** ein Seed, explorativ, die Anlauflänge nicht gesucht. In B10 lief **keine** Form mit Anlauf, auch TAT nicht. Für einen fairen Vergleich muss TAT dann ebenfalls mit Anlauf laufen.

## Folge für V1 (Vorschlag)

**Präregistrierte Studie V1, „Anlauf statt kleiner LR“:**
- **Einzige Änderung gegenüber B10:** linearer Anlauf über 5 % der Schritte. Der Anlauf gilt für **alle** Formen (BPreCap, LB, TAT), damit der Vergleich fair bleibt.
- **Neues Tuning:** LR-Raster 1e-4, 2e-4, 5e-4 und WD 0, 0,3, 3,0 je Form, 2 Seeds, dann 12 Seeds in der Bestätigung.
- **Primärtest:** Wird BPreCap mit Anlauf bei seiner getunten LR besser als ohne, und hält der 4-Bit-Vorteil gegen TAT, wenn beide mit Anlauf laufen?
- **Kosten** auf dem M1 Pro: etwa 54 Tuning-Läufe à 2–4 min, dazu 36 Bestätigungsläufe, zusammen etwa 3–4 h mit 2 Workern.
- **Vorab-Erwartung:**
  - 0,75, dass BPreCap mit Anlauf bei 2e-4 nicht mehr einbricht;
  - 0,5, dass der 4-Bit-Vorteil gegen TAT erhalten bleibt, weil TAT mit Anlauf vielleicht ebenfalls besser wird.

**Vertrauen:**
- 0,9, dass die Hülle nicht die Ursache des Einbruchs ist (LB ohne Sättigung, harter Clip gleich);
- 0,7, dass der fehlende LR-Anlauf die Hauptursache ist (ein Seed, sehr deutlich).
