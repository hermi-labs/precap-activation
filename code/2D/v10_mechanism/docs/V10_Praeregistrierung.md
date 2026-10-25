# V10 Präregistrierung — Mechanismusstudie nach V9

Datum: 21.10.2026
Status: geplant, noch nicht gestartet.

## Ausgangslage
V9 ergab kein monotones Geometriegesetz. Die stärkste Verbesserung von `precap_s225` lag bei `a=0.25`, während `a=0.75` deutlich schwächer bzw. teils schlechter ausfiel. Damit ist die nächste Frage nicht mehr primär **ob** `s225` manchmal hilft, sondern **welcher Mechanismus** diesen Unterschied erklärt.

## Primärziel
Vergleich der internen Dynamik von `precap_base` und `precap_s225` in zwei gezielt gewählten Aufgabenregimen:
- Erfolgsfall: `geo_a025`
- Gegenfall: `geo_a075`

jeweils bei:
- Tiefe 16
- Tiefe 32
- 20 neuen Seeds

## Modelle
- `precap_base`
- `precap_s225`

Optional nur kontextuell, nicht primär inferenzrelevant:
- `silu`
- `gelu`

## Messgrößen
Zusätzlich zu train/test MSE auf festen Checkpoints (z. B. 0, 50, 100, 200, 400, 800, 1200):

1. Pro Schicht Aktivierungs-Quantile: p01, p10, p50, p90, p99
2. Pro Schicht Pre-Activation-Quantile
3. Sättigungs-/Kappungsanteil pro Schicht für PreCap-Varianten
4. Gradienten-Normen pro Schicht (oder pro Gewichtsmatrix)
5. Anteil sehr kleiner Aktivierungen / tote Bereiche als Kontrastmaß
6. Endpunktdiagnostik auf Validierungsgrid: Fehlerkarten oder mindestens Ortsquantile des Fehlers

## Primärhypothesen
H1. Auf `geo_a025` zeigt `precap_s225` gegenüber `precap_base` frühere oder gleichmäßigere Signalweitergabe über mittlere und tiefe Schichten.

H2. Auf `geo_a075` zeigt `precap_s225` keinen solchen Vorteil bzw. eine weniger günstige interne Verteilung.

H3. Der beobachtete Leistungsunterschied ist stärker mit internen Verteilungs- und Gradientenmustern verknüpft als mit einem einfachen globalen Geometrieskalar.

## Primäre Auswertung
1. Endpunktvergleich `geo_a025` vs `geo_a075` für `precap_s225 - precap_base` wie in V9 als reine Replikationskontrolle.
2. Für jede Schicht und jeden Checkpoint robuste Medianprofile über Seeds.
3. Vordefinierte Fokusmaße:
   - Verteilungsausbreitung in tiefen Schichten
   - Anteil gekappter Aktivierungen
   - Gradientenerhalt in mittleren/tiefen Schichten
4. Keine neue globale Behauptung ohne reproduzierbares Muster in mindestens beiden Tiefen oder mit klar begründetem Tiefenunterschied.

## Entscheidungslogik
V10 ist primär erklärend, nicht selektiv. Es soll keinen neuen Standardparameter küren, sondern einen belastbaren Mechanismus identifizieren oder die Mechanismushypothese verwerfen.

## Erwartete Konsequenz
- Falls ein konsistenter Mechanismus sichtbar wird, folgt danach V11 als gezielte bestätigende Studie.
- Falls kein konsistenter Mechanismus sichtbar wird, bleibt `precap_base` Standard und `s225` eine aufgabenspezifische Sondervariante ohne verallgemeinerbare Regel.
