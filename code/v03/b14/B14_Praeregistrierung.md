# B14 – Präregistrierung: Feste weiche Hülle gegen lernbaren harten Clip (PACT-Stil) im TAT-Rezept (CIFAR-10, d100, privat)

Stand: 31.10.2026, spätabends, vor dem Start. PreCap, die Tiefenfamilie und Lagma sind Aarons Ideen. BPreCap (= kL2B) ist ein Arbeitsname. Das Design habe ich mit Claude entworfen. Aaron hat B14 (PACT) am 31.10. gegen 21:45 für die Nacht freigegeben („Du kannst PACT auch noch machen … nichts veröffentlichen“).

## Ausgangslage

- **PACT** (Choi et al. 2018, [arXiv 1805.06085](https://arxiv.org/pdf/1805.06085)) ist der bekannteste Ansatz aus der Quantisierung, Aktivierungen zu begrenzen.
  - Die Grenze α ist lernbar, gelernt per Gradient mit einer L2-Strafe auf α.
  - Das Original ist einseitig (ReLU, clip(x, 0, α)) und für Quantisierung während des Trainings gedacht.
- Unsere Form hat eine **feste**, **weiche** Grenze. Ein naheliegender Einwand gegen v0.3: „Why not just learn the clipping bound like PACT?“
- B13 vergleicht weich gegen hart bei fester Grenze. B14 ändert gegenüber B13 nur eines: Die harte Grenze wird lernbar.

## Frage

Ist BPreCap (feste weiche Hülle, M = 3) im TAT-Rezept auf CIFAR-10 (d100) bei fairen 4 Bit besser als derselbe Kern mit einem harten Clip an einer lernbaren Grenze je Schicht?

## Eine Änderung gegenüber B13: die harte Grenze ist lernbar

- **kL2Bp_tr:** PreCap-Kern wie kL2Bh_tr, y = clip(Kern(z), −|α_i|, |α_i|).
  - Je Schicht ein trainierbares α_i, Start bei 3,0. Bei der Init ist das Netz damit identisch zu kL2Bh_tr, im Mechanik-Test bitgleich geprüft.
- **LBp_tr:** linearer Kern wie LBh_tr, Start bei M = 2,299.
- **Symmetrisch:** Unsere Netze sind ungerade, der Kern hat negative Werte. Das Original-PACT ist einseitig. Die symmetrische Form ist die direkte Übertragung.
- **Gradient nach α** per Autograd: +1, wo der Kern über α liegt, −1, wo er unter −α liegt, sonst 0. Das entspricht PACT. Der Betrag |α| hält die Grenze positiv.
- **Strafe auf α:** Der entkoppelte Gewichtsabfall von AdamW wirkt auch auf α und zieht die Grenze nach unten, wie die L2-Strafe bei PACT. Seine Stärke wird mit dem WD-Raster getunt.
- **Regime, Tuning und Messung wie B13:** LR-Spitze {5e-5, 1e-4, 2e-4, 5e-4} × WD {0; 0,3; 3,0}, 2 Tuning-Seeds, frühes Stoppen, Quantisierung nach dem Training mit derselben Kalibrierung.
- **Vergleichsformen** (gepaart, Seeds 42168–42179):
  - kL2B_tr und LB_tr aus B10 (fest, weich)
  - kL2Bh_tr und LBh_tr aus B13 (fest, hart)

## Läufe

- Tuning 2 × 12 × 2 = 48.
- Bestätigung 2 × 12 = 24.
- **Kontrolle:** kL2B_tr, Seed 42168, mit der B10-Wahl, bitgleich zu B10.
- Zusammen 73 Läufe, etwa 1,7 h. Start nach B13, nur am Netzteil.

## Primärtest

**kL2B_tr (B10) − kL2Bp_tr bei w8a4a_best am Punkt des frühen Stoppens.** Wilcoxon zweiseitig exakt, gepaart, α = 0,05.

**Erfolg:** p < 0,05 und Median < 0.

## Sekundär (Holm über 7)

| Nr. | Vergleich |
|---|---|
| S1 | LB_tr − LBp_tr, w8a4a |
| S2 | kL2B_tr − kL2Bp_tr, fp32 |
| S3 | LB_tr − LBp_tr, fp32 |
| S4 | kL2B_tr − kL2Bp_tr, w8a4 |
| S5 | LB_tr − LBp_tr, w8a4 |
| S6 | kL2Bp_tr − kL2Bh_tr (B13), w8a4a. Hilft das Lernen der Grenze? |
| S7 | LBp_tr − LBh_tr (B13), w8a4a |

**Beschreibend:**
- gelernte Grenzen am Ende (Median, Minimum, Maximum über die Schichten)
- Bereich, 4-Bit-Preis, Stopp-Punkte, gewählte LR und WD

## Deutungsregeln (vorab)

| Ergebnis | Deutung |
|---|---|
| Erfolg | Die feste weiche Hülle ist bei 4 Bit besser als ein gelernter harter Clip. Antwort auf „why not PACT?“ |
| Primär „unklar“ | Bei 4 Bit gleichauf mit einem gelernten Clip. Vorteil von BPreCap dann nur: kein zusätzlicher Parameter, kein Tuning der Strafe |
| Primär „schlechter“ | Ein gelernter harter Clip ist bei 4 Bit besser. Für v0.3 offen berichten und die Aussage zu K1 einschränken |
| S6 „besser“ | Das Lernen der Grenze hilft der harten Form bei 4 Bit |

## Vorab-Erwartung

| Größe | Erwartung |
|---|---|
| Primär „besser“ | 0,4 |
| Primär „unklar“ | 0,45 |
| Primär „schlechter“ | 0,15 |
| S6 „besser“ (Lernen hilft der harten Form) | 0,4 |
| WD-Wahl > 0 bei mindestens einer PACT-Form | 0,7 |
| Kontrolle bitgleich | 0,95 |

## Grenzen

- **Nachbildung, nicht das Original-PACT:** symmetrisch, auf unseren kritischen Kernen, Strafe über AdamW statt eigener L2-Strafe. PACT war für Quantisierung während des Trainings gedacht, hier wird nach dem Training quantisiert. Beides gilt für alle Formen gleich, wird aber als Grenze genannt.
- **Beweglichkeit der Grenze unter Adam:** Adam ändert jeden Parameter je Schritt höchstens etwa um die aktuelle LR. Über den Kosinus-Lauf (Summe der LR = Spitze × 3440) kann sich α deshalb höchstens um etwa 0,34 (Spitze 1e-4) bis 1,72 (Spitze 5e-4) bewegen.
  - Die Grenze kann sich also nur begrenzt von ihrem Startwert lösen. Beim Original-PACT mit SGD war das anders.
  - Wird so berichtet. Liegt die gelernte Grenze am Ende nahe am Start, misst B14 vor allem „harte Grenze nahe M“.
- Nur 2 Tuning-Seeds, nur d100, CIFAR-10.
- Die Vergleichsläufe aus B10 waren beim Entwurf bekannt, die aus B13 nicht (B13 läuft erst nach dem Freeze von B14).

## Freeze

Eingefroren werden `b14_run.py`, `b14_analyze.py`, `test_b14_mechanik.py`, `test_b14_analyse.py`, diese Datei, `hp_from_b10.csv`, `src/precap/*.py` und die CIFAR-10-Dateien.

Vorab gemachte Tests (31.10., spätabends):

- **Selbsttest:** n_confirm 25, n_tune 48. Gain und χ1 der PACT-Formen gleich den harten Formen.
  - Der Wert `mlx_vs_numpy` ist bei den PACT-Formen groß. Das ist erwartet: `make_act` liefert dort nur den Kern, die Grenze setzt `Net.post`. Die volle Form prüft der Mechanik-Test.
- **Auswertung an künstlichen Daten** (`test_b14_analyse.py`): AUSWERTUNG OK.
  - Erfolg und „schlechter“ werden erkannt, ebenso S6 mit den harten Formen aus B13 als dritter Quelle. Null mit 20 Wiederholungen: 1 Fehlalarm.
  - Eine falsche Kontrolle wird erkannt. Ein Ausfall zählt als 90 %. Der relative Pfad findet B10.
- **Mechanik** (`test_b14_mechanik.py`): **MECHANIK OK** (31.10., 21:41, 34 Prüfungen). Das Startskript führt den Test erneut aus und bricht bei Fehler ab. Geprüft wird:
  - Bei der Init ist die Vorwärtsrechnung bitgleich zur harten Form.
  - 100 trainierbare Grenzen, alle mit Gradient ungleich 0.
  - Die Kalibrierung sieht die Grenze.
  - Die Grenze ändert sich beim Training. In 172 Schritten bei LR 1e-4: 2,988–2,994 statt 3,0.
  - Gewichtsabfall 3,0 zieht sie stärker nach unten (2,92).
  - Kontrolle bitgleich zu B10. Die Zwischenmessung verändert das Training nicht.
- **Rauchtest:** im Startskript.
