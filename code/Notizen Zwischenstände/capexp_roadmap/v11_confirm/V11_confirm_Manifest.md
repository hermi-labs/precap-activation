# CapExp v11 confirm — vorab festgelegtes Manifest

Festgeschrieben am 20.10.2026 vor dem ersten Run, nach der v11-mini-Auswertung (kein neuer Kandidat bestand Gate B; `hybrid_gate` Knie 0,75, beta 0,3 war der knappe Fall).

## Kandidat und Referenzen

- Kandidat: `hybrid_gate` mit c=1, a=1,25, gamma=0,01, s=1,5, Knie 0,75, beta 0,3.
- Referenzen: `precap` (c=1, a=1,25, gamma=0,01, s=1,5), SiLU, GELU.
- `precap_wg` entfällt (im Mini klar schlechter als PreCap).

## Raster (2400 Runs)

- Aufgaben: `sin3_quad`, `sin_mix`, `abs_wave`; 3000 Schritte Adam; gleicher Datengenerator wie im Mini.
- Tiefen 4, 8, 16: `plain` und `res_inv`, je 3 Paare (init_k, lr): (0,5; 0,001), (1,0; 0,001), (1,0; 0,003).
- Tiefen 32, 64: nur `res_inv`, ein Paar (0,5; 0,001); jetzt auch mit SiLU und GELU.
- Seeds 100–109 (frisch; die Mini-Seeds 0–2 wurden zur Kandidatenwahl benutzt).
- Reihenfolge seed-weise, damit auch ein vorzeitiger Abbruch vollständig gepaarte Seeds liefert.

## Metrik

d = log10(MSE_Kandidat / MSE_Vergleich), gepaart je Aufgabe, Tiefe, Architektur, init_k, lr, Seed. d < 0 heißt Kandidat besser. Zusätzlich: Median d, Seed-Siege, Stabilitätsmetriken (hmax_peak, cap_final, grad_final).

## Entscheidungsregel (vor dem Lauf fixiert)

Der Kandidat besteht, wenn gegen PreCap gilt:

1. Tief (Tiefe 32 und 64, 6 Zellen): in jeder Zelle Median d <= log10(1,5) = 0,176.
2. Flach (Tiefe 4, 8, 16; 18 Zellen): in mindestens 11 von 18 Zellen Median d < 0, gepoolter Median d < 0 und Win-Rate > 50 %.
3. Stabilität: keine numerischen Ausfälle (bad=False) und kein Anstieg von hmax_peak um mehr als Faktor 2 gegenüber PreCap im Median.

Alternativ besteht er, wenn er tief reproduzierbar besser ist (mindestens 4 von 6 tiefen Zellen Median d < 0 und gepoolte Win-Rate >= 2/3).

Besteht er nicht, bleibt PreCap die beste getestete Variante; die Daten werden trotzdem als Mechanismus-Evidenz dokumentiert.
