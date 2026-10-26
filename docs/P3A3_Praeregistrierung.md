# Phase 3a-3 – Präregistrierung (χ₁-angepasste SPreCap-Familie)

Stand: 26.10.2026, vor dem Start eingefroren. Hashes: `code/phase3/p3a3_chi1/meta/frozen_sha256.txt`.

## Anlass

3a-2 hat einen klaren Nachteil von SPreCap gegenüber SELU bestätigt (4 von 4 Zellen schlechter). Das Gradientenprofil beim Start passt dabei gut zur Mean-Field-Rechnung: Gemessen wurde \(\log_{10}(g_1/g_L)\), vorhergesagt ist \(\tfrac{L-1}{2}\log_{10}\chi_1\).

| Funktion | χ₁ | Tiefe 100: vorhergesagt | Tiefe 100: gemessen | Tiefe 50: vorhergesagt | Tiefe 50: gemessen |
|---|---|---|---|---|---|
| SPreCap | 1,1305 | 2,64 | 2,77 / 2,81 | 1,31 | 1,33 / 1,36 |
| SELU | 1,0716 | 1,49 | 1,54 / 1,56 | 0,73 | 0,69 / 0,71 |

Hypothese: SPreCap ist wegen des größeren χ₁ zu "chaotisch". Die Gradienten zur Eingabe hin sind bei Tiefe 100 etwa 600-fach größer als am Ausgang, bei SELU etwa 35-fach. Die gleiche Funktion mit χ₁ = SELU sollte sich deshalb wie SELU verhalten. Die Konstanten sind exakt mit `math/p3a3_chi1_family.py` berechnet.

## Aufbau

- **Netz und Training:** wie 3a-2. MLP mit Breite 256, Tiefe 50 und 100, ohne Normierung und ohne Residual. Adam, Batch 128, 1720 Schritte, Checkpoints 0/430/860/1290/1720. Datensätze MNIST und FashionMNIST mit fester Validierung (die letzten 5000). Init \(W\sim N(0,g^2/\text{fan\_in})\) mit varianzerhaltendem g.
- **LR-Raster:** wie 3a-2, also 1e-5, 2e-5, 5e-5, 1e-4, 2e-4 und 5e-4. In 3a-2 lag für SPreCap, SELU und ELU keine gewählte LR am Rand.
- **Aktivierungen (6):**

| Name | λ | α | b | χ₁ | Wertebereich |
|---|---|---|---|---|---|
| spreccap | 0,80676869644 | 2,39695130922 | 1 | 1,1305 | [−1,50; 2,48] |
| spreccap_cs | 1,102702500 | 2,053776267 | 0,71188112 | 1,0716 | [−1,76; 3,39] |
| spreccap_e03 | 1,985552225 | 1,608136397 | 0,4118896262 | 1,0300 | [−2,48; 6,11] |
| spreccap_cs_ls | wie cs | wie cs | wie cs | 1,0716 beim Start | s je Schicht lernbar |
| selu | Standard | | | 1,0716 | unbeschränkt |
| elu | Standard | | | – | unbeschränkt, beste Funktion in 3a-2 |

  Alle SPreCap-Varianten haben die Form \(f(x)=\lambda\,p(bx)\) für \(x>0\) und \(\lambda\alpha\,p(bx)\) für \(x\le 0\), mit \(p(u)=\mathrm{CapExp}(s\tanh(u/s))\), s = 1,5 und c = 1. Bei `spreccap_cs_ls` ist \(s=1{,}5\,e^{\rho}\) je Schicht, mit \(\rho=0\) beim Start und einer eigenen festen Adam-LR von 1e-2. Mit der Haupt-LR würde sich s in 1720 Schritten kaum bewegen.
- **Seeds:** Tuning 42000–42003 (576 Runs), Bestätigung 42010–42024 (360 Runs), zusammen 936 Runs.
- **LR-Wahl:** höchste mediane val_acc beim letzten Schritt über die Tuning-Seeds. Divergierte Runs zählen mit 0,1. Bei Gleichstand gewinnt die kleinere LR.

## Tests (Bestätigungs-Seeds, Testfehler beim letzten Schritt, gepaart nach Seed)

- **Statistik:** exakter Vorzeichen-Permutationstest der Wilcoxon-Rangsumme, zweiseitig. Nullen werden entfernt, bei Bindungen gelten Mittelränge. Grund für den Wechsel: In 3a-2 lieferten scipy 1.13 auf dem Mac und 1.18 in der Sandbox bei Bindungen unterschiedliche p-Werte. Die Urteile waren gleich, der neue Test ist aber versionsunabhängig. Der scipy-Wert wird nur zur Information mitgeschrieben.
- **Primär:** spreccap_cs gegen selu in 4 Zellen, Holm über 4, α = 0,05.
  - Vorteil bestätigt: mindestens 3 von 4 Zellen signifikant besser und keine signifikant schlechter.
  - Nachteil bestätigt: das Spiegelbild.
- **Entscheidungsregel für das Zwischenziel v0.1:** spreccap_cs gilt als bewährt, wenn es in keiner Zelle signifikant schlechter ist als SELU (Holm primär) und in mindestens 3 von 4 Zellen die mediane Differenz höchstens +0,5 Prozentpunkte beträgt.
- **Sekundär:** eigene Holm-Familie über 24 Tests, gleiche Regel:
  - cs gegen spreccap (hilft das angepasste χ₁?)
  - cs_ls gegen cs (hilft lernbares s?)
  - cs gegen elu
  - e03 gegen selu
  - e03 gegen cs
  - spreccap gegen selu (Wiederholung von 3a-2 mit neuen Seeds)
- **Deskriptiv:**
  - Testfehler über χ₁
  - Gradientenprofil beim Start, gemessen gegen vorhergesagt
  - gelerntes s (min/med/max je Zelle)
  - LR-Fenster, Divergenzrate, mittlerer Rang

## Erwartungen vor dem Start

| Frage | Erwartung | Konfidenz |
|---|---|---|
| Gradientenprofil von cs liegt bei Tiefe 100 im Bereich 1,49 ± 0,15 | ja | 0,8 |
| cs ist besser als spreccap (mindestens 3 von 4 Zellen) | ja | 0,65 |
| cs bewährt sich gegen SELU (Entscheidungsregel) | nein | 0,6 |
| Nachteil von cs gegen SELU bestätigt | nein | 0,55 |
| cs_ls unterscheidet sich nicht bestätigt von cs | ja | 0,6 |
| ELU bleibt beste Funktion nach mittlerem Rang | ja | 0,6 |
| Wiederholung spreccap gegen selu: Nachteil wieder bestätigt | ja | 0,8 |

## Ablauf

`launch_p3a3.sh` wie in 3a-2: caffeinate, Lock, Hash-Prüfung, Selbsttest, 7 Worker Tuning, Integrität, LR-Wahl, 7 Worker Bestätigung, Integrität, Analyse, `done.txt`. Geschätzte Laufzeit etwa 3,5–4 Stunden.
