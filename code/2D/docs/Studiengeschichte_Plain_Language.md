# CapExp / PreCap — Studiengeschichte in normaler Sprache

## Worum es die ganze Zeit ging
PreCap ist deine Aktivierungsfunktion. In den bisherigen 2D-Studien war sie oft stark, aber nicht in jeder Konfiguration gleich stabil. Vor allem bei tiefen Netzen entstand die Kernfrage: **Warum ist PreCap manchmal klar besser, aber nicht immer gleich zuverlässig?**

Es ging also nicht nur um "mehr Leistung", sondern um die Ursache hinter den Unterschieden: Tiefe, Aufgabenstruktur, Optimierung, Aktivierungsdynamik oder Zufall über Seeds.

## Frühere Versionen: Erst die grobe Leistung klären
Die frühen Versionen prüften zuerst, ob PreCap überhaupt ein ernsthafter Kandidat ist. Ergebnis: PreCap-base war auf mehreren 2D-Aufgaben konkurrenzfähig und teils besser als Standardaktivierungen, besonders in tieferen MLPs.

Danach kam die erste Ursachenfrage: Liegt das an stärkerer oder schwächerer Begrenzung? Die Antwort war nein, jedenfalls nicht einfach. Mehr Deckelung half nicht allgemein, und auch ein einzelner offensichtlicher numerischer Defekt erklärte das Muster nicht.

## V7a und V7b: Ist es Tiefe oder s?
Mit V7a und V7b wurde gezielt geprüft, ob ein größeres s wie `s225` das Tiefenproblem löst. Das Ergebnis war gemischt: Auf manchen Aufgaben half `s225` real, aber nicht stabil genug über alle Seeds und nicht als allgemeiner Fix.

Wichtig war die Einsicht: **Die Tiefe allein erklärt den Effekt nicht.** Auf `radial_ripple` sah `s225` besser aus als auf `ridge_mix`; damit rückte die Aufgabenstruktur in den Vordergrund.

## V8: Aufgabenstruktur statt nur Tiefe
V8 testete genau diese Vermutung mit radialen und ridge-artigen Aufgabenfamilien. Dort zeigte sich ein klares Muster: `s225` half über die radialen Aufgaben hinweg deutlich eher als über die ridge-artigen.

Das war wichtig, weil die Forschung sich damit von blindem Parametersuchen wegbewegte. Statt "welcher Wert ist global besser?" wurde die Frage: **Für welche Geometrie oder Aufgabenart hilft `s225` überhaupt?**

## V9: Kontinuum statt harte Schubladen
V9 machte den nächsten sauberen Schritt. Statt Aufgaben grob in zwei Gruppen einzuteilen, wurde ein kontrollierter Übergang von radial zu ridge aufgebaut und mit einer Anisotropie-Zahl beschrieben.

Das Ergebnis war aufschlussreich: Es gibt zwar einen Zusammenhang zwischen Aufgabenstruktur und `s225`-Effekt, aber **nicht als einfache lineare Regel**. Der stärkste Vorteil lag nicht am rein radialen Ende, sondern bei einer Mischgeometrie (`a=0.25`). Damit ist die einfache Hypothese "je radialer, desto besser s225" nicht bestätigt.

## Was heute als gesichert gilt
- `precap_base` bleibt die robuste Standardwahl.
- `s225` ist kein allgemeiner Ersatz, aber auch kein Zufallseffekt; es hilft bei bestimmten Aufgabenstrukturen real.
- Das ursprüngliche Tiefenproblem ist nicht durch einen einzelnen Parameterwert gelöst.
- Die Ursache hängt mit Aufgabenstruktur zusammen, aber nicht monoton und nicht simpel.

## Warum wir jetzt nicht blind weiter sweepen sollten
Noch mehr c- oder s-Werte würden aktuell vor allem neue Zahlen produzieren, aber wenig Erklärung. Nach V9 ist der sinnvollste nächste Schritt deshalb **Mechanismusforschung**: Was passiert in den Aktivierungen, Gradienten und internen Darstellungen genau dann, wenn `s225` hilft — und was anders, wenn es nicht hilft?

## Nächster sauberer Schritt: V10
V10 sollte nicht wieder ein großer Suchlauf sein, sondern eine gezielte Mechanismusstudie. Der Fokus liegt auf dem starken V9-Fall (`a=0.25`) gegen den problematischen Gegenfall (`a=0.75`), jeweils bei Tiefe 16 und 32, mit `precap_base` gegen `s225` und zusätzlichen Schichtdiagnostiken.

Damit wird aus der bisherigen Ursachenforschung ein echter Test auf **warum** der Effekt entsteht.
