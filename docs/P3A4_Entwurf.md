# Phase 3a-4 – Entwurf: kritische orthogonale Init („Schumacher-Setup“)

Stand: 26.10.2026. Dies ist ein Entwurf, noch keine Präregistrierung. Eingefroren wird erst, wenn 3a-3 ausgewertet ist. So kann das Ergebnis von 3a-3 nicht nachträglich in dieses Design einfließen.

## Ziel

Wir wollen beide Ideen fair weiter prüfen, also PreCap (PreCap0) und SPreCap (Variante cs). Zwei Fragen stehen dabei im Mittelpunkt:

1. Lernt eine Funktion in tiefen Netzen (Tiefe 100), wenn man sie genau an der Stabilitätsgrenze startet? Das heißt orthogonale Gewichte und χ1 = 1 („edge of chaos“).
2. Bleibt sie über einen breiteren Bereich von Lernraten stabil als SELU? Das ist die Robustheitsfrage, die durch die tanh-Begrenzung beantwortet werden soll.

## Mathematische Grundlage

Das Skript ist `math/p3a4_kritischer_gain.py`, die Ausgabe steht in `.out`. Gerechnet wird exakt mit quad, der Knick bei 0 wird getrennt integriert.

| Funktion | Gain varianzerhaltend | χ1 dabei | kritischer Gain (χ1=1) | q* dort | Ableitungs-Streuung κ dort |
|---|---|---|---|---|---|
| PreCap0 | 1.0120 | 1.0637 | 0.9524 | 0.146 | 0.346 |
| SPreCap-cs | 1.0000 | 1.0716 | 0.7986 | 0.0013 | 0.350 |
| SPreCap | 1.0000 | 1.1305 | 0.6838 | 0.0006 | 0.469 |
| SELU | 1.0000 | 1.0716 | 0.7002 | 0.0006 | 0.211 |
| ELU | 1.2452 | 1.0359 | 1.0134 | 0.0012 | 0.0015 |

κ = E[f′⁴]/E[f′²]² − 1. Bei κ = 0 sind alle Schichtableitungen gleich, das ist die Voraussetzung für dynamische Isometrie ([Pennington 2017](https://research.google/pubs/resurrecting-the-sigmoid-in-deep-learning-through-dynamical-isometry-theory-and-practice/)).

Was die Tabelle zeigt:

- **SELU, SPreCap und ELU:** Für sie liegt der kritische Punkt bei einem Signal von fast null (q* ≈ 0.001). Das Netz startet dort praktisch linear.
- **ELU:** Ist um 0 glatt, mit Steigung 1 auf beiden Seiten. Deshalb erreicht ELU dort fast perfekte Isometrie (κ ≈ 0.002) und ist die stärkste Kontrolle.
- **SELU und SPreCap:** Behalten ihren Knick, die Steigungen links und rechts sind verschieden. Beim kritischen Punkt bleibt κ deshalb zwischen 0.21 und 0.47.
- **PreCap0:** Ist die einzige Funktion mit einem echten Arbeitspunkt (q* = 0.146). Allerdings liegt dieser Punkt direkt an einer Sattel-Knoten-Stelle der Varianzabbildung. Schon bei g = 0.95 springt q* auf 0. Der Arbeitspunkt ist also empfindlich, und das ist ein wichtiger Vorbehalt.

## Explorativer Vorabtest (nicht präregistriert)

Die Bedingungen waren: Sandbox, 1 Seed (42000), MNIST, Tiefe 100, Breite 256, Adam, 600 Schritte, orthogonale Init. Skript und Rohwerte liegen in `code/phase3/p3a4_vorab/`.

| Funktion | Init | log10(g1/gL) | val acc bei LR 1e-4 | val acc bei LR 5e-4 |
|---|---|---|---|---|
| PreCap0 | varianzerhaltend | 0.86 | 0.106 (tot) | 0.106 |
| PreCap0 | kritisch | 0.54 | **0.830** | 0.106 |
| SPreCap-cs | varianzerhaltend | 1.56 | 0.936 | 0.106 |
| SPreCap-cs | kritisch | 0.30 | 0.907 | 0.201 |
| SELU | varianzerhaltend | 1.56 | 0.938 | 0.097 |
| SELU | kritisch | 0.30 | 0.902 | 0.144 |
| ELU | varianzerhaltend | 0.41 | 0.950 | 0.099 |
| ELU | kritisch | 0.31 | 0.941 | 0.124 |

Vorsichtige Lesart, weil nur 1 Seed und ein kurzer Lauf vorliegen:

- **PreCap0:** Die kritische Init holt PreCap0 bei LR 1e-4 aus dem Totalausfall zurück, von Zufallsniveau auf 83 %. Das ist der erste klare Hinweis, dass PreCap0 nicht an der Funktion selbst scheitert, sondern am Startpunkt.
- **Gradientengefälle:** Bei allen Funktionen sinkt das Gefälle über die Tiefe stark, etwa von 1.56 auf 0.30 Größenordnungen. Die Genauigkeit nach 600 Schritten steigt dadurch aber nicht, sondern fällt leicht. Ein fast linearer Start lernt anfangs langsamer.
- **LR 5e-4:** Alle Funktionen sterben, unabhängig von der Init. Ein Robustheitsvorteil zeigt sich hier noch nicht. SPreCap-cs kritisch liegt mit 0.20 knapp vorn, aber 1 Seed sagt dazu nichts.

## Geplantes Design (Entwurf)

- **Funktionen:** precap0, spreccap_cs, selu, elu, tanh. tanh ist die klassische Isometrie-Referenz.
- **Init:** orthogonal, je zwei Gains: varianzerhaltend und kritisch, Werte aus der Tabelle oben. Für tanh gilt g = 1. Das ergibt 10 Konfigurationen.
- **Daten und Tiefe:** MNIST und Fashion-MNIST, Tiefe 100. Breite 256, Batch 128, 1720 Schritte wie in 3a-2 und 3a-3.
- **Lernraten:** 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3.
- **Tuning und Robustheit:** 6 Seeds (42100–42105) über das ganze LR-Gitter. Das sind 5 × 2 × 7 × 2 × 6 = 840 Läufe.
- **Bestätigung:** 15 Seeds (42110–42124) bei der besten LR. Das sind 5 × 2 × 2 × 15 = 300 Läufe.
- **Laufzeit:** Gesamt 1140 Läufe, auf dem Mac geschätzt 5–6 h. Alternativ geht es schneller auf dem RTX-4070-Rechner, das wäre zugleich der erste Praxistest für den CUDA-Kernel.

### Hypothesen (Entwurf)

- **H1 (Init-Effekt, primär):** PreCap0 kritisch gegen PreCap0 varianzerhaltend, Fehler bei der besten LR, je Datensatz.
- **H2 (Robustheit, primär):** Fläche unter der LR-Kurve, also die mittlere val acc über alle 7 LRs je Seed. Verglichen werden gepaart über Seeds:
  - SPreCap-cs gegen SELU bei gleicher Init
  - PreCap0 kritisch gegen SELU kritisch
- **H3 (sekundär):** Bestes Ergebnis von PreCap0 kritisch und SPreCap-cs kritisch gegen ELU kritisch.
- **Zusatzmaß:** die höchste LR, bei der der Median noch mindestens 90 % der eigenen besten val acc erreicht.
- **Statistik:** wie bisher, also exakter gepaarter Vorzeichen-Permutationstest und Holm je Hypothesenfamilie.

### Erwartungen (vorläufig)

| Aussage | Wahrscheinlichkeit |
|---|---|
| PreCap0 kritisch ist besser als PreCap0 varianzerhaltend | 0.8 |
| PreCap0 kritisch erreicht SELU (gleiche Init) | 0.3 |
| SPreCap-cs ist robuster über LR als SELU | 0.35 |
| ELU bleibt insgesamt vorn | 0.65 |

## Danach

Der nächste Schritt ist der Test jenseits von MNIST: ein kleiner Transformer auf Tiny Shakespeare und CIFAR-10, außerdem 3b mit weniger LayerNorm. Ob PreCap dabei weiterkommt, hängt von 3a-3 und 3a-4 ab. Ist das Ergebnis eindeutig negativ, wird es ehrlich dokumentiert und gesichert. Auch das ist ein Ergebnis.
