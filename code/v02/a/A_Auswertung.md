# Idee A – Auswertung Tiefen-PreCap χ1 = 1 + k/L

*Veröffentlichte Fassung vom 29.10.2026: Die Ergebnisse dieser Studie sind unverändert. Entfernt wurden der Vermerk „privat“, Querverweise auf noch unveröffentlichte Studien und ein Abschnitt, der A mit einer solchen Studie vergleicht. Präregistrierung und Skripte sind die eingefrorenen Originale.*

Stand: 27.10.2026, abends. Die Studie lief von 13:07 bis etwa 17:50 auf dem Mac M1 Pro (MLX).

## Ergebnis in Kürze

- **Urteil laut Präregistrierung:** kL1 und kL2 sind beide **"besser als SELU"**. Alle 8 Primärzellen (Tiefe 100 und 200, MNIST und Fashion) sind signifikant besser, jede mit 15 von 15 Seeds. Damit ist Idee A erfolgreich.
- **Datenlage:** 432/432 Tuning-Runs und 450/450 Bestätigungs-Runs, keine Ausfälle. Die Hashes aller eingefrorenen Dateien sind OK.
- **Tiefe 100:** Der Vorteil ist realistisch groß.
  - MNIST: 4,56 % (kL1) gegen 6,46 % (SELU)
  - Fashion: 15,51 % gegen 16,64 %
- **Tiefe 200:** SELU und cs werden in 1720 Schritten gar nicht fertig trainiert. Ihr Trainingsverlust liegt bei etwa 0,9–1,3, der Testfehler bei 32–40 %. Die kL-Varianten erreichen 5,4–6,0 % (MNIST) bzw. 16,5–17,1 % (Fashion). Die riesigen Differenzen von bis zu −34 pp messen also vor allem die **Trainingsgeschwindigkeit bei gegebenem Budget**, nicht die erreichbare Endgenauigkeit.
- **Preis:** Die 4-Bit-Robustheit geht verloren. Der Wertebereich wächst auf 7–14, und w8a4_best wird ab Tiefe 100 um 13–41 pp schlechter. Bei w8a8 gibt es keinen Unterschied. Das war vorab als Zielkonflikt benannt.
- **Einordnung:** Funktionen tiefenabhängig "linearer" zu machen ist **bekanntes Prinzip**:
  - Deep Kernel Shaping (Martens et al. 2021, https://arxiv.org/abs/2110.01765)
  - Tailored Activation Transformations (Zhang, Botev, Martens 2022, https://arxiv.org/abs/2203.08120)
  - shaped activations mit Steigung 1 + c/√n (Li et al. 2022, https://proceedings.neurips.cc/paper_files/paper/2022/hash/45fc4a0da7e7f6fbabaabe2d20a441d1-Abstract-Conference.html)

  Neu ist nur die konkrete Umsetzung: eine beschränkte, selbstnormierende Aktivierung, deren χ1 über eine einzige Konstante b exakt auf 1 + k/L gesetzt wird. Ein Vergleich mit TAT/DKS fehlt noch. Ohne ihn darf nichts davon öffentlich als Vorteil behauptet werden.

## Primärtest: kL gegen SELU (fp32-Testfehler, Wilcoxon, Holm über 8)

| Variante | Zelle | Median-Diff. | Seeds besser/schlechter | Holm-p | Urteil |
|---|---|---|---|---|---|
| kL1 | MNIST 100 | −1,76 pp | 15/0 | 0,0005 | besser |
| kL1 | Fashion 100 | −1,36 pp | 15/0 | 0,0005 | besser |
| kL1 | MNIST 200 | −33,87 pp | 15/0 | 0,0005 | besser |
| kL1 | Fashion 200 | −15,71 pp | 15/0 | 0,0005 | besser |
| kL2 | MNIST 100 | −1,43 pp | 15/0 | 0,0005 | besser |
| kL2 | Fashion 100 | −0,83 pp | 15/0 | 0,0005 | besser |
| kL2 | MNIST 200 | −33,64 pp | 15/0 | 0,0005 | besser |
| kL2 | Fashion 200 | −15,38 pp | 15/0 | 0,0005 | besser |

0,0005 ist das kleinstmögliche Holm-p bei 15 Paaren und 8 Tests.

## Medianwerte

| Zelle | Aktivierung | LR | Testfehler | Δ w8a4_best | max. Aktivierung | Gradientengefälle gemessen / vorhergesagt |
|---|---|---|---|---|---|---|
| MNIST 50 | cs | 5e-5 | 4,53 % | 2,1 | 3,35 | 0,82 / 0,74 |
| MNIST 50 | e03 | 5e-5 | 3,96 % | 3,4 | 5,83 | 0,44 / 0,31 |
| MNIST 50 | kL1 | 1e-4 | 4,09 % | 1,6 | 7,92 | 0,35 / 0,21 |
| MNIST 50 | kL2 | 1e-4 | 4,22 % | 1,2 | 4,89 | 0,48 / 0,42 |
| MNIST 50 | SELU | 1e-4 | 4,29 % | 1,3 | 7,60 | 0,83 / 0,74 |
| Fashion 50 | cs | 1e-4 | 14,61 % | 1,2 | 3,37 | 0,75 / 0,74 |
| Fashion 50 | e03 | 5e-5 | 14,27 % | 4,1 | 5,81 | 0,42 / 0,31 |
| Fashion 50 | kL1 | 5e-5 | 14,21 % | 5,2 | 7,50 | 0,36 / 0,21 |
| Fashion 50 | kL2 | 5e-5 | 14,42 % | 3,3 | 4,79 | 0,50 / 0,42 |
| Fashion 50 | SELU | 1e-4 | 14,98 % | 2,0 | 7,67 | 0,77 / 0,74 |
| MNIST 100 | cs | 5e-5 | 6,31 % | 2,9 | 3,34 | 1,67 / 1,49 |
| MNIST 100 | e03 | 5e-5 | 5,06 % | 3,6 | 5,87 | 0,69 / 0,64 |
| MNIST 100 | kL1 | 5e-5 | **4,56 %** | 17,9 | 11,14 | 0,35 / 0,21 |
| MNIST 100 | kL2 | 2e-5 | 5,04 % | 24,3 | 6,96 | 0,45 / 0,43 |
| MNIST 100 | SELU | 5e-5 | 6,46 % | 5,0 | 6,18 | 1,67 / 1,49 |
| Fashion 100 | cs | 5e-5 | 16,95 % | 3,0 | 3,34 | 1,64 / 1,49 |
| Fashion 100 | e03 | 5e-5 | 16,64 % | 4,7 | 5,89 | 0,68 / 0,64 |
| Fashion 100 | kL1 | 2e-5 | **15,51 %** | 30,5 | 10,08 | 0,39 / 0,21 |
| Fashion 100 | kL2 | 2e-5 | 15,62 % | 22,9 | 7,01 | 0,48 / 0,43 |
| Fashion 100 | SELU | 5e-5 | 16,64 % | 5,2 | 6,26 | 1,64 / 1,49 |
| MNIST 200 | cs | 2e-5 | 40,04 % | 31,6 | 3,30 | 3,22 / 2,99 |
| MNIST 200 | e03 | 2e-5 | 8,73 % | 24,6 | 5,70 | 1,41 / 1,28 |
| MNIST 200 | kL1 | 2e-5 | 5,97 % | 70,3 | 14,21 | 0,50 / 0,22 |
| MNIST 200 | kL2 | 2e-5 | **5,44 %** | 71,7 | 10,32 | 0,50 / 0,43 |
| MNIST 200 | SELU | 2e-5 | 39,53 % | 34,5 | 5,21 | 3,23 / 2,99 |
| Fashion 200 | cs | 2e-5 | 32,99 % | 32,0 | 3,30 | 3,22 / 2,99 |
| Fashion 200 | e03 | 1e-5 (Rand) | 19,44 % | 43,7 | 5,42 | 1,38 / 1,28 |
| Fashion 200 | kL1 | 1e-5 (Rand) | 17,05 % | 61,3 | 11,73 | 0,50 / 0,22 |
| Fashion 200 | kL2 | 2e-5 | **16,53 %** | 64,5 | 10,31 | 0,53 / 0,43 |
| Fashion 200 | SELU | 2e-5 | 32,19 % | 37,8 | 5,16 | 3,24 / 2,99 |

Das Gradientengefälle ist log10(‖∇W₁‖/‖∇W_L‖) bei der Init, die Vorhersage ist (L−1)/2·log10 χ1.

## Sekundärtests

- **S1, kL gegen cs:** in allen 12 Tests besser. Bei Tiefe 50 sind es −0,35 bis −0,59 pp, bei Tiefe 100 −1,3 bis −1,7 pp.
- **S2, kL gegen e03:**
  - Tiefe 50: kein Unterschied.
  - Tiefe 100: kL1 besser (−0,41 / −1,21 pp), kL2 nur bei Fashion besser.
  - Tiefe 200: kL1 in beiden Zellen besser (−3,1 / −2,7 pp), kL2 bei MNIST besser (−3,7 pp), bei Fashion knapp nicht signifikant (Holm 0,050).
  - Das tiefenabhängige χ1 bringt also gegenüber dem festen χ1 = 1,03 vor allem bei Tiefe 200 etwas.
- **S3, Tiefe 50 gegen SELU:** nur kL1 bei Fashion besser (−0,59 pp), sonst unklar.
- **S4, feste Varianten gegen SELU:**
  - cs: überall unklar. Das wiederholt 3a-3, cs ist gleichauf mit SELU.
  - e03: besser bei Fashion 50, MNIST 100, MNIST 200 und Fashion 200. Das bestätigt den 3a-3-Befund bei MNIST 100 mit neuen Seeds.
- **S5, Quantisierung:**
  - w8a8: praktisch keine Unterschiede.
  - w8a4_best: kL ist ab Fashion 50 bzw. Tiefe 100 überall deutlich schlechter als SELU und als cs.
  - cs gegen SELU bei w8a4_best wiederholt Q1b ein weiteres Mal: besser bei Fashion 50, MNIST 100, Fashion 100 und Fashion 200, schlechter bei MNIST 50.
- **Gradiententheorie:** Die Rangfolge stimmt in allen 6 Zellen exakt mit der Vorhersage überein. Bei cs, e03 und SELU liegt die Messung 0,05–0,24 über der Vorhersage, bei kL1 0,14–0,28. Das ist ein konstanter Offset durch die endliche Breite und die Ausgabeschicht, die Skalierung stimmt.
- **LR am Rand:** kL1 und e03 wählen bei Fashion 200 die kleinste LR 1e-5. Mit noch kleinerer LR könnten beide eventuell besser werden. Das ändert das Urteil für kL1 nicht.

## Abgleich mit der Vorab-Erwartung

| Aussage | Erwartung | eingetreten |
|---|---|---|
| kL2 "besser als SELU" | 0,35 | ja |
| kL1 "besser als SELU" | 0,25 | ja |
| Erfolg A | 0,4 | ja |
| kL1 bei Tiefe 200 schlechter als kL2 | 0,6 | knapp ja (5,97 gegen 5,44 % und 17,05 gegen 16,53 %, nicht getestet) |
| Rangfolge des Gefälles stimmt | 0,8 | ja |

## Grenzen

- **Budget:** Fester Trainingsumfang von 1720 Schritten. Bei Tiefe 200 ist das ein Geschwindigkeitsvergleich. Ob SELU mit mehr Schritten aufholt, ist offen.
- **Baselines:** Es fehlen die stärksten bekannten Verfahren für tiefe Netze ohne Normierung und Residuals, nämlich TAT (Leaky ReLU) und DKS. Ohne diesen Vergleich gibt es keine öffentliche Aussage "besser als der Stand der Technik".
- **Aufbau:** Nur MLPs mit Breite 256, MNIST und Fashion.
- **Quantisierung:** kL gibt den Vorteil "beschränkt" praktisch auf. Für 4-Bit ist kL ungeeignet, dort bleibt cs die richtige Wahl.

## Nächste Schritte (Vorschlag)

1. **A2:** kL2 gegen TAT (Leaky ReLU, offizielle DeepMind-Implementierung https://github.com/google-deepmind/dks) und SELU bei Tiefe 100/200.
   - Zwei Budgets: 1720 Schritte und 4× so viele.
   - Bei MNIST 200 zusätzlich ein Lauf bis zur Konvergenz.
   - Diese Frage entscheidet, ob A veröffentlichbar ist.
2. **Danach:** kritische Init (3a-4) zusammen mit kL, dann B (Netze mit Normierung) und D.
3. **Öffentlich:** vorerst nichts, bis A2 steht.

## Dateien

- Daten: `workers/`, `tune/`, `lr_tuned.csv`
- Auswertung: `analysis/a_result.json`, erzeugt mit `a_analyze.py` (scipy) in der Sandbox
- Präregistrierung: `A_Praeregistrierung.md` (eingefroren 13:07)
