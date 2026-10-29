# Danksagung und Werkzeuge

## Persönliche Worte

Die Idee am 14.10. war folgende: Warum gibt es keine exponentielle Funktion, die nach oben unbegrenzt wächst und sich gleichzeitig gut regeln lässt?

Ich habe bewusst auf andere Beispiele verzichtet, um die Entwicklung nicht in eine bestimmte Richtung zu lenken. Doch wie fängt man an, wenn man zunächst keine Ahnung hat, wie man eine solche Idee praktisch und schnell in Code/ Simulationen etc umsetzt? Die Grundidee war bereits vorhanden, aber erst mithilfe von Opus und Sonnet nahm sie konkrete Formen an. Das Modell half mir dabei, meine Gedanken weiterzuentwickeln und die notwendigen Skripte zu programmieren, weil meine Programmiererfahrung mit Python nicht so tief war.

Gemeinsam haben wir die Höhen und Tiefen dieses Prozesses durchlebt – stets mit dem klaren Ziel vor Augen, die Formel zu finden, von der wir schon in der S-Bahn geträumt hatten.

Mein Mac und Perplexity liefen dabei auf Hochtouren. Opus und auch Sonnet waren eine enorme Unterstützung bei der Umsetzung meiner Ideen in Code. Was zunächst nur in meinem Kopf existierte, begann sich Schritt für Schritt in praktikable Formeln zu verwandeln. Aus ersten Ansätzen entstanden neue Erkenntnisse, die wiederum weitere Ideen usw.

Für diese Hilfe danke ich.

## Verwendete Werkzeuge

Die Idee zu CapExp und PreCap, die Fragestellungen und alle Entscheidungen stammen von Aaron Schmitt. Er hat die Experimente auf eigener Hardware durchgeführt (MacBook Pro M1 Pro, RTX 4070).

Mitgearbeitet hat Perplexity Computer, ein KI-Assistent ([perplexity.ai](https://www.perplexity.ai)) primär mit den Modellen Claude Opus 5.5 und Claude Sonnet 5.5 von Anthropic. Er hat mitgewirkt an:

- mathematischen Herleitungen
- Skripten, Präregistrierungen und statistischen Auswertungen
- den Kernels
- Texten

Alle Ergebnisse wurden mit festgelegten Hashes und reproduzierbaren Skripten geprüft.

## Version 0.2

Die Fragen für v0.2 (Tiefe, Vergleich mit dem Stand der Technik, Quantisierung) und die Entscheidungen stammen wieder von Aaron Schmitt. Die Studien liefen auf seinem MacBook Pro M1 Pro.

Ab v0.2 hat Claude Code von Anthropic ([claude.com/claude-code](https://claude.com/claude-code)) mitgearbeitet, mit dem Modell Claude Opus 5.5. Mit Claude entstanden:

- Entwürfe der Präregistrierungen und die Studienskripte (A, A2, A3, Q1, Q1b, Q2, Q3)
- die Auswertungen und Berichte
- der Text und die Abbildungen des Preprints v0.2
- die Literaturprüfung

Aaron hat die Studien freigegeben, teils einzeln, teils als Rahmen für Folgestudien, die Claude danach begründet ausgewählt hat. Alle Präregistrierungen wurden vor dem Lauf per SHA-256 eingefroren und sind unverändert im Repo.
