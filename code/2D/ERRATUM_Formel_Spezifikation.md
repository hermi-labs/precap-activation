# ERRATUM — PreCap-Formel in der Spezifikation

Datum: 20.10.2026

Betroffene Datei: v2_holdout/docs/PreCap_Urformel_Spezifikation.md (bleibt unveraendert als historisches Dokument).

Problem: Die Datei nennt PreCap(x) = x - 0.01*softplus((x-2)/1.5) mit Steigung zwischen 0.993 und 1. Alle Runner (v1b, v2, v3, v5, v6) implementieren stattdessen
  z=a*x; u=z+gamma*z^2; v=s*tanh(u/s); PreCap = CapExp(v)
mit CapExp(v) = exp(v)-1 (v<=c) bzw. exp(c)-1+exp(c)*(v-c) (v>c).

Geprueft: In jedem Runner steht der Selbsttest gegen genau diese Form (formula_max_abs_err unter 1e-6, in v6 = 0.0).

Konsequenz: Die Ergebnisse v1b bis v6 gelten fuer die tanh+CapExp-Form. Die Softplus-Beschreibung (inkl. Ableitungsintervall und Beschreibung "kein harter Cap") trifft darauf nicht zu und darf nicht zitiert werden. Insbesondere ist PreCap in der Implementierung fuer kleine x etwa exp(x)-1-artig, nicht identitaetsnah.

Empfehlung: Spezifikation als v10-Text neu schreiben, bevor irgendetwas extern verwendet wird.
