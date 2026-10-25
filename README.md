# PreCap / CapExp – Aktivierungsfunktionen für tiefe Netze

Forschungsprojekt von Aaron Schmitt, begonnen am 14.10.2026.

- **CapExp:** exponentiell bis zu einer Schwelle c, danach tangential linear. Mathematisch entspricht das \(e^c\cdot\mathrm{ELU}(x-c)+(e^c-1)\).
- **PreCap:** CapExp mit vorgeschalteter weicher Begrenzung \(v = s\tanh((z+\gamma z^2)/s)\), \(z = a x\).

Status: in Arbeit, privat. Siehe `docs/PROJEKT_GESAMTDOKUMENTATION.md` und `docs/PreCap_Fahrplan_Veroeffentlichung.md`.

## Struktur

- `docs/` – Gesamtdokumentation, Fahrplan, Quell-Dokumente der Studien
- `notizen/` – handschriftliche Notizen (Matrix-Idee, 16.10.2026)
- `code/` – Skripte, Präregistrierungen, Hashes und Auswertungen aus 1D, 2D und 3D (wird ergänzt)

Rohdaten der Läufe liegen lokal und sind nicht Teil des Repositorys.
