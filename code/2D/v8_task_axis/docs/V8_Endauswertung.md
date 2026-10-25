# V8 Endauswertung — Aufgabenachse (21.10.2026)

Status: gültig abgeschlossen. 960/960 Runs, 5760/5760 Zeilen, eindeutige Schlüssel, 0 NaN/inf, gefrorene Hashes unverändert.

## Primärentscheidung
Die Präregistrierung verlangte für einen Radial-Hinweis: mindestens 4/6 radialen Aufgabe×Tiefe-Zellen mit negativem Median d=log10(MSE_s225/MSE_base) und negativer Median der Zellmediane. Für einen Gruppen-Kontrast: radial >=4/6 negativ, ridge <=2/6 negativ.

| Gruppe | negative Zellen | Median der Zellmediane | Entscheidung |
|---|---:|---:|---|
| radial | 6/6 | -0.121 | Radial-Hinweis bestätigt |
| ridge | 4/6 | -0.021 | strenger Gruppen-Kontrast nicht bestätigt |

Der vorab definierte Richtungskontrast ist somit nicht formal bestätigt, weil ridge mehr als zwei negative Zellmediane hat. Die radiale Komponente ist hingegen vollständig konsistent.

## Zellbefunde s225 vs base
| Aufgabe | d16 Median d / Siege / p | d32 Median d / Siege / p |
|---|---|---|
| radial_bessel | -0.288 / 18/20 / 0.00085 | -0.322 / 14/20 / 0.0148 |
| radial_ripple | -0.151 / 17/20 / 0.00116 | -0.083 / 14/20 / 0.0348 |
| radial_highfreq | -0.009 / 10/20 / 0.184 | -0.090 / 12/20 / 0.0825 |
| ridge_curved | -0.119 / 14/20 / 0.00859 | -0.021 / 11/20 / 0.215 |
| ridge_mix | -0.022 / 12/20 / 0.406 | -0.080 / 13/20 / 0.392 |
| ridge_highfreq | +0.037 / 8/20 / 0.715 | +0.048 / 8/20 / 0.959 |

Nach Zell-Bonferroni (p<0.0083) bestehen radial_bessel d16 und radial_ripple d16; ridge_curved d16 verfehlt mit p=0.00859 knapp.

## Explorativ (nicht primär)
Radial gepoolt: Median -0.129, 85/120 Siege, p=9.36e-8; seed-gemittelt 20/20, Median -0.181, p=9.54e-7. Ridge gepoolt: Median -0.022, 66/120, p=0.364; seed-gemittelt 9/20, Median +0.021, p=0.676. Der gepaarte Seed-Kontrast radial minus ridge ist -0.156 (18/20, p=3.15e-5). Das stützt die Aufgabenabhängigkeit deutlich, obwohl das strengste primäre Zellanzahl-Kriterium knapp verfehlt wurde.

## Referenzen (explorativ)
Radial: base schlägt SiLU in 114/120 und GELU in 112/120; s225 schlägt SiLU in 119/120 und GELU in 118/120. Ridge: base schlägt SiLU in 118/120 und GELU in 104/120; s225 schlägt SiLU in 118/120 und GELU in 106/120. s225 ist bei radial gegenüber den Referenzen klar stärker, bei ridge nicht.

## Schluss
s225 ist kein universeller Ersatz für base. Die Evidenz stützt einen Nutzen für radiale/isotrope Aufgaben, besonders radial_bessel und radial_ripple, während ridge_highfreq klar dagegen spricht. Die verbleibenden leicht negativen ridge-Zellen verhindern eine harte binäre Trennung; die gruppengepaarte Evidenz spricht dennoch stark für einen radialen Vorteil.
