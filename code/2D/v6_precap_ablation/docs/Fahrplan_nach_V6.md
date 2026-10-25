# Fahrplan nach V6 (21.10.2026)

## Leitfrage
Ist die Tiefe-64-Schwäche ein Ausreißer-/Varianzproblem der Optimierung oder ein Parameterproblem?
V6 spricht stark für Varianz.

## Schritte
1. **v7a Seed-Erweiterung:** radial_ripple d64, ridge_mix d64, Anker radial d16; Varianten base, c125, s225, SiLU, GELU; ca. 20 neue Seeds (Zahl vor Start einfrieren). Auswertung vorab festlegen: Median, getrimmter Mittelwert, Wilcoxon gepaart, Ausreißerrate (Test-MSE > 5x Median), Mindestsiegquote. Keine Änderung der Basisformel.
2. **Ausreißer-Mechanismus:** Verläufe der schlechten Seeds (radial d64 Seed 900, Seed 901 Spike; ridge d64 Seed 901) mit Snapshots wiederholen; prüfen: Initialisierung, Lernrate, Gradientenspitzen, Layer-Lokalisierung.
3. **Optional feinere c/s-Skala:** nur wenn v7a eine konsistente Richtung zeigt (z. B. s 1.5–2.5, c 1.0–1.5).
4. **Adaptives c/s:** zurückgestellt, bis ein fester Wert robust gewinnt.
5. **Brücke zu 1D:** Vergleich, warum 1D-Matrizen den Effekt deutlich stärker zeigen (Optimierung, Skalierung, Tiefe, Breite).

## Abbruchregeln
- Keine neue Variante während einer Auswertung hinzufügen.
- Jede neue Studie: Formel-Selfcheck, Manifest, Hashes, Kriterien vor Start einfrieren.
