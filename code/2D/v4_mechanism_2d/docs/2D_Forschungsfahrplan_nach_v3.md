# CapExp — 2D-Forschungsfahrplan nach v3

**Status:** Planungsdokument. Es startet keine neue Studie und verändert weder die v2- noch die v3-Daten, Formel oder Bewertung.

## Ausgangslage

v2 und v3 bestätigen ein replizierbares Signal von PreCap auf den drei vorab definierten Kernaufgaben. Die Evidenz reicht über die große Replikation bis Tiefe 32. Tiefe 64 markiert jedoch eine belastbare Grenze: Der aggregierte Effekt wird klein und ist gegen GELU leicht ungünstig. Die nächste Phase ist deshalb Mechanismus- und Grenzforschung, nicht erneute Suche nach einem beliebigen positiven Effekt.

## Arbeits-Hypothese

> Unter dem eingefrorenen Trainingsprotokoll verbessert PreCap auf bestimmten strukturhaltigen 2D-Aufgaben die Test-MSE, vermutlich teilweise über kontrolliertere Aktivierungs- und Gradienten-Dynamik. Der Nutzen ist aber abhängig von Tiefe und Aufgabenstruktur und verliert in sehr tiefen Netzen an Robustheit.

## Phase A — Mechanismus-Snapshots

**Frage:** Treten niedrigere Aktivierungspeaks und Gradienten zeitlich und schichtweise vor einem Generalisierungsvorteil auf?

**Design:** Keine neue Formel, keine neue Optimierung. Gezielt gespeicherte Snapshots bei Schritt 0, 100, 500, 1.500 und 3.000; zusätzlich Layer-Statistiken für Pre-Activation, Post-Activation, Skip-Anteil, Peak, Mittelwert, Standardabweichung und Gradienten-Norm.

**Gezielte Zellen:**

| Klasse | Aufgabe | Tiefe | Warum |
|---|---|---:|---|
| Klarer Gewinner | radial_ripple | 16 | Stabiles positives Signal |
| Klarer Gewinner | angled_kink | 32 | Neue Aufgabenfamilie, starkes Signal |
| Tiefer Grenzfall | radial_ripple | 64 | Effekt kippt bei großer Tiefe |
| Tiefer Grenzfall | ridge_mix | 64 | Effekt kippt bei großer Tiefe |
| Referenz-Gegenbeispiel | highfreq_grid | 32 | Nahezu neutral bzw. leicht ungünstig |

**Vergleich:** Für jede Zelle PreCap, SiLU und GELU mit denselben Seeds und Datenströmen. Priorität haben wenige, sorgfältig analysierte Paare statt ein großer neuer Sweep.

**Auswertung:**

- Verlauf von Test-MSE, Trainings-MSE, globaler Gradienten-Norm und Aktivierungspeak
- Layerweise Quantile/Peaks der Vor- und Nachaktivierungen
- Zeitlicher Zusammenhang: Stabilitätsänderung vor/nach dem Auseinanderlaufen der Testfehler
- Testziel: Mediationshinweis, keine Kausalitätsbehauptung

## Phase B — Tiefe-64-Grenzdiagnose

**Frage:** Ist die Grenze bei 64 primär Optimierung, Architektur oder Aufgabenstruktur?

**Kontrolliertes Minimaldesign:**

| Achse | Fixiert | gezielte Variation |
|---|---|---|
| Formel | PreCap-v9, unverändert | Keine Formeländerung |
| Aufgabe | radial_ripple, ridge_mix, corner_kink | Vergleich mit highfreq_grid als Gegenbeispiel |
| Tiefe | 32 und 64 | Fokus auf Differenz statt breitem Sweep |
| Architektur | Res-Inv, Breite 64 | Optional eine zweite Breite erst nach Phase A |
| Optimierung | Adam, LR 0.001, 3.000 Schritte | Optional Lernkurvenanalyse, keine nachträgliche Suche |
| Seeds | frische, vorab notierte Seeds | 5 für Diagnose, später 10 zur Bestätigung |

**Entscheidungslogik:** Erst beobachten, ob Layer-Sättigung, wachsende Varianz oder Gradientendynamik den Knick zwischen 32 und 64 begleitet. Erst danach eine einzige gezielte Interventionsachse auswählen.

## Phase C — Aufgabenstruktur-Karte

**Frage:** Welche Eigenschaften einer Zieloberfläche sagen einen PreCap-Vorteil voraus?

**Vordefinierte Strukturachsen:**

- Glattheit/Krümmung: glatt bis knickartig
- Frequenz: niedrig bis hoch
- Orientierung: achsenparallel bis schräg
- Globalität: global periodisch bis lokal stückweise
- Mischstruktur: Kombination aus glatt, periodisch und Knick

**Designprinzip:** Ein kleines faktorielles Aufgabenraster mit dokumentierter Formel je Aufgabe, nicht nach Ergebnis ausgewählte Aufgaben. Jede Familie erhält mehrere Stärkestufen statt nur einen Einzelfall.

**Erfolg:** Eine reproduzierbare, vorab definierte Strukturassoziation; kein Anspruch auf universelle Generalisierung.

## Phase D — Snapshot-Explorer

**Ziel:** Kein Live-Monitoring im Training. Stattdessen nach Phase A gespeicherte Snapshots interaktiv ansehen.

**Viewer-Inhalte:**

- Auswahl: Aufgabe, Tiefe, Aktivierung, Seed, Trainingsschritt
- 2D-Outputfläche und Fehlerfläche
- Layerweise Aktivierungsverteilung und Peak-Verlauf
- Ansicht einzelner, ausgewählter Neuronen über das Eingabegitter
- Gegenüberstellung PreCap versus SiLU/GELU

**Warum offline:** Das verhindert Logging-Overhead während des Trainings und macht die Darstellung reproduzierbar sowie hardwarefreundlich.

## Reihenfolge und Stop-Regeln

1. Phase A zuerst; ohne Mechanismus-Signatur keine Architektur- oder Formeländerungen.
2. Phase B nur mit den dokumentierten Grenzfällen beginnen.
3. Phase C erst nach klarer Mechanismus- oder Grenzbeobachtung.
4. Phase D parallel zur Auswertung von Phase A vorbereiten, nicht als Voraussetzung für neue Runs.
5. Wenn Phase A keine stabile Verbindung zwischen Dynamik und Effekt zeigt, Hypothese auf reine empirische Strukturabhängigkeit zurückstufen.

## Unverhandelbare Regeln

- PreCap-v9 bleibt als Referenzformel unverändert.
- Alle neuen Seeds werden vor Start notiert.
- Jede neue Hauptfrage bekommt ein eigenes Manifest, einen Selfcheck, Hashes und eine eingefrorene Auswertungsregel.
- Explorative Befunde bleiben klar getrennt von bestätigenden Entscheidungen.
