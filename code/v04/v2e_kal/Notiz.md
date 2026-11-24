# V2e-Nachmessung mit fester Kalibrierung (explorativ, 20.11.2026, 02:29–03:29)

Nicht präregistriert, mit Claude. Die gespeicherten V2e-Gewichte (BPreCap und ctanh, Seeds 42350–42352) wurden mit der eingefrorenen Messfunktion `v2e_run.measure` bei mse4, max und p99999 gemessen. Kontrolle: BPreCap 42350 mse4 ist identisch mit der V2e-Messung (True). Auch die drei ctanh-Messungen mit der damals gewählten Kalibrierung stimmen mit V2e überein.

Tabelle: `kal_tabelle.txt` (Wert BPreCap/ctanh je Seed, q = BPreCap/ctanh). Rohdaten: `mess/`, Log: `logs/kal.log`.

Befund:
- Bei gleicher Kalibrierung rundet ctanh feiner. r̄ ist bei BPreCap in 9 von 9 Zellen um 8–14 % größer, passend zum engeren Bereich von ctanh.
- Die Fehler der Schichten addieren sich bei ctanh in 9 von 9 Zellen gleichgerichteter (Superposition q 0,81–0,99).
- Bei mse4 (abschneidend) gibt ctanh die Rundungsfehler viel stärker weiter (ḡ^Q q 0,75–0,86, 3/3). Bei den weiten Kalibrierungen ist die Weitergabe ähnlich, ctanh sogar etwas schwächer.
- Gesamtstörung D^Q: Bei mse4 ist BPreCap 3/3 kleiner (q 0,78–0,92), bei max und p99999 je 2/3.
