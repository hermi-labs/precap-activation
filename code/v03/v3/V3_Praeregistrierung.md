# V3 – Präregistrierung: QAT als Obergrenze – feste weiche Hülle mit PTQ gegen LSQ- und PACT-QAT (CIFAR-10, d100, TAT-Rezept, privat)

Stand: 03.11.2026, vormittags, vor dem Start. Die Freeze-Zeit steht in `meta/frozen_at.txt`. PreCap, BPreCap und Lagma sind Aarons Ideen. Das Design ist mit Claude entworfen und wird in Aarons Abwesenheit gestartet (Leitlinie fürs Alleinarbeiten, Freigabe vom 03.11. bis 18:00). Teil V3 des Fahrplans v0.4. Den Anstoß gab ein Reddit-Kommentar zu v0.3: „compare against PACT and LSQ with QAT as an upper bound“.

## Ausgangslage

- **v0.3 und V1:** BPreCap ist bei 4 Bit nach dem Training (PTQ) besser als TAT. Mit Anlauf sind es −8,82 pp (V1, 12/0).
- **Bisher nur PTQ:** Alle Vergleiche liefen mit PTQ. In der Praxis wird für 4 Bit oft mit Quantisierung im Training gearbeitet (QAT).
- **Der PACT-Versuch in B14 war kein fairer Test:** Die Grenze lernte unter AdamW nicht, weil die gemeinsame LR zu klein war, es kein QAT gab und WD statt L2 wirkte.
- **Offen ist deshalb:**
  - Ist BPreCap mit einfacher PTQ so gut wie TAT mit gängigem QAT?
  - Hilft die feste Hülle auch, wenn beide Formen mit QAT trainiert werden?

## Ablauf je (Form, Seed)

1. **Basisnetz:** wie die V1-Bestätigung, auf denselben Seeds 42168–42179. Anlauf 5 %, Kosinus, 6880 Schritte, V1-Wahl:
   - BPreCap: kL2B_tr, LR 1e-4, WD 0,3;
   - TAT: LR 2e-4, WD 0.

   **Pflichtprüfung:** val, test, w8a4a_best und gewählte Kalibrierung müssen bitgleich zur V1-Zeile sein.
2. **Drei Feintunings aus derselben Basis**, je 1376 Schritte (20 %):
   - **Gewichte:** AdamW mit 0,1 × LR-Spitze, Kosinus auf 0, gleicher WD.
   - **Batch-Reihenfolge:** für alle drei gleich (Seed + 999).

   | Variante | Was | 4-Bit-Messgröße |
   |---|---|---|
   | **FT** | Feintuning ohne Quantisierung. Das ist die Kontrolle für das Budget | PTQ w8a4a_best am feingetunten Netz, Protokoll wie V1 |
   | **LSQ** | QAT. Die Eingänge aller 101 Linearschichten werden 4 Bit asymmetrisch fake-quantisiert, mit lernbarem Schritt s und lernbarem Nullpunkt z (LSQ/LSQ+). Gradientenskalierung g = 1/√(N·15), N = Merkmale am Eingang. Gewichte int8 je Ausgangskanal mit STE | Testfehler mit den gelernten Quantisierern |
   | **PACT** | QAT mit lernbaren Grenzen [lo, hi] je Schicht. Gradient nur aus geclippten Werten, L2-Strafe 2·10⁻⁴ auf lo und hi. Gewichte wie LSQ | wie LSQ |

   - **Start der Quantisierer:** exakt das PTQ-Gitter der V1-Wahlregel am Basisnetz. Am Start ist der QAT-Fehler bitgleich zum PTQ-Fehler, das prüft der Mechanik-Test.
   - **Optimierer der Quantisierer:** eigener Adam ohne WD, LR_Q mit Kosinus auf 0.
3. **Tuning von LR_Q:**
   - **Raster:** {1e-4, 1e-3, 1e-2}, je Form und Methode.
   - **Tuning-Seeds:** 42000 und 42001. Die Basis wird dort bitgleich zum V1-Tuning gerechnet, geprüft über den Val-Fehler.
   - **Wahl:** kleinster medianer Val-Fehler mit QAT-Inferenz; bei Gleichstand die kleinere LR_Q; ein Ausfall zählt als 90 %.

## Läufe

- **Tuning:** 2 Formen × 2 Seeds × (Basis, FT, 3 × LSQ, 3 × PACT) = 32 Zeilen.
- **Bestätigung:** 2 Formen × 12 Seeds × (Basis, FT, LSQ, PACT) = 96 Zeilen.
- **Dauer:** geschätzt 1,5–2 h auf 2 Workern (M1 Pro), nur am Netzteil.

## Primärtest

**BPreCap-FT (PTQ) − TAT-LSQ (QAT)**, 4-Bit-Testfehler, gepaart je Seed. Wilcoxon zweiseitig exakt, 12 Paare, α = 0,05. Ein Ausfall zählt als 90 %.

## Sekundär (Holm über 6)

| Nr. | Vergleich | Frage |
|---|---|---|
| S1 | BPreCap-FT (PTQ) − TAT-PACT (QAT) | wie primär, gegen PACT |
| S2 | BPreCap-LSQ − TAT-LSQ | Hilft die Hülle auch, wenn beide QAT nutzen? |
| S3 | BPreCap-PACT − TAT-PACT | dito mit PACT |
| S4 | TAT-LSQ − TAT-FT (PTQ) | Was bringt QAT bei TAT? |
| S5 | BPreCap-LSQ − BPreCap-FT (PTQ) | Was bringt QAT bei BPreCap? |
| S6 | BPreCap-FT − TAT-FT (beide PTQ) | Wiederholung von V1 nach gleichem Feintuning |

**Beschreibend:**
- fp32 je Variante;
- 4-Bit-Kosten;
- wie weit sich die Quantisierer-Grenzen bewegen (Median der relativen Breitenänderung). Das zeigt, ob PACT diesmal lernt;
- PTQ nach QAT;
- Tuning-Tabelle;
- gewählte LR_Q.

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Primär „besser“ | Mit einfacher PTQ ist BPreCap besser als TAT mit LSQ-QAT. Die feste Hülle ersetzt hier QAT. Das ist eine starke Aussage für v0.4, nur für MLPs |
| Primär „unklar“ | Mit PTQ erreicht BPreCap das Niveau von TAT mit LSQ-QAT, ohne Quantisierung im Training |
| Primär „schlechter“ | QAT ist besser. Der Vorteil von BPreCap ist ein PTQ-Vorteil. Was bleibt, entscheidet S2: Ist S2 „besser“, hilft die Hülle auch unter QAT. Ist S2 „unklar“, schließt QAT den Abstand |
| PACT-Grenzen bewegen sich kaum (relative Breite unter 1 %) | PACT hat wieder nicht gelernt. S1 und S3 gelten dann nicht als fairer PACT-Test. Das wird offen gesagt |
| Basis nicht bitgleich | Abweichung suchen und berichten. Die Tests gelten trotzdem, weil die Paarung innerhalb von V3 liegt |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Basis bitgleich (24 von 24) | 0,9 |
| Primär „besser“ | 0,2 |
| Primär „unklar“ | 0,35 |
| Primär „schlechter“ | 0,45 |
| S2 „besser“ (Hülle hilft auch unter QAT) | 0,5 |
| S4 „besser“ (QAT hilft TAT) | 0,85 |
| S5 „besser“ (QAT hilft BPreCap) | 0,6 |
| PACT-Grenzen bewegen sich um mehr als 1 % | 0,7 |

## Grenzen

- **QAT nur als Feintuning:** 20 % der Schritte aus einem fp32-Netz, wie bei LSQ und PACT üblich. Kein QAT von Anfang an.
- **Feste Einstellungen:** Die LR der Gewichte im Feintuning (0,1 × Spitze) ist fest. Getunt wird nur LR_Q.
- **PACT:** asymmetrisch mit zwei Grenzen, an den Eingängen der Linearschichten. Das Original sitzt hinter ReLU mit einer Grenze. Die L2-Stärke ist fest.
- **LSQ:**
  - Der Nullpunkt ist im Training kontinuierlich, wie bei LSQ+, nicht ganzzahlig.
  - Es gibt keinen Gradienten durch den Rundungsfehler selbst, nur STE.
- **Rahmen:** nur MLPs, CIFAR-10, d100, TAT-Rezept, 4-Bit-Aktivierungen mit int8-Gewichten, simuliert.
- **Messzeitpunkt:** Gemessen wird am Ende des Feintunings, ohne Auswahl eines Checkpoints.

## Freeze und Tests

**Eingefroren werden:**
- `v3_qat.py`, `v3_analyze.py`;
- die Tests;
- diese Datei;
- `../v1/v1_run.py`, gegen den V1-Freeze geprüft.

**Vorab gemachte Tests:**
- **Selbsttest:** ok.
  - LSQ und PACT liefern am Start exakt die PTQ-Rundung, maximale Differenz 0.
  - Es gibt 16 Stufen, die Gradienten sind endlich und ungleich 0.
  - Alle Vergleichszeilen in V1 wurden gefunden.
- **Mechanik:** MECHANIK OK.
  - Das Basisnetz tat 42168 ist über volle 6880 Schritte bitgleich zu V1.
  - QAT am Start entspricht exakt PTQ, für LSQ und für PACT.
  - PACT bekommt keinen Gradienten aus Werten im Bereich, wohl aber aus geclippten. LSQ bekommt einen Gradienten zu s.
  - Mit LR_Q = 0 bleiben die Quantisierer fest, die Gewichte ändern sich. Mit 1e-2 bewegen sich die PACT-Grenzen.
- **Auswertung an künstlichen Daten:** AUSWERTUNG OK. Szenarien:
  - QAT besser, BPreCap besser;
  - Null (20×, höchstens 3 Fehlalarme; gemessen 1);
  - Ausfall, Basis nicht bitgleich.
- **Rauchtest:** 60 Basisschritte und 20 Feintuning-Schritte, beide Stufen, 16 Zeilen ok.
