import csv,math,statistics as st,os
from collections import defaultdict
R=list(csv.DictReader(open('v2_holdout_results.csv')));M=defaultdict(dict);S=defaultdict(dict)
for r in R:
 k=(r['task'],int(r['depth']),int(r['seed']));M[k][r['act']]=float(r['test_loss']);S[k][r['act']]={z:float(r[z]) for z in ['hmax_peak','grad_final','cap_final']}
def eff(rows,ref):return [math.log10(m['precap']/m[ref]) for _,m in rows]
def cell(rows,ref):
 x=eff(rows,ref);return f'{st.median(x):+.3f} ({sum(y<0 for y in x)}/{len(x)})'
T=['radial_ripple','ridge_mix','corner_kink'];D=[4,8,16]
L=['# CapExp v2 — lebende Evidenz- und Bewertungstabelle','', '> **Status:** Diese Tabelle wird aus dem eingefrorenen Ergebnisdatensatz `v2_holdout_results.csv` erzeugt. Sie trennt bestätigte Evidenz, Grenzen und die vorab festgelegten Kriterien für Folgestudien. Aktualisierung: `python update_evidence_scorecard.py`.','', '## Evidenzstatus','','| Evidenzfrage | Ergebnis | Bewertung |','|---|---|---|',f'| Vollständigkeit | {len(R)}/270 Runs; {sum(r["bad"]=="True" for r in R)} bad | Gültig |',f'| Replikation des v1b-Pilots | Neue Seeds 400–409, 2/3 Aufgaben gewinnen gegen SiLU und GELU | Bestätigt |', '| Formelkonstanz | v9-Referenz, `formula_max_abs_err=0` vor Start | Bestätigt |', '| Generalisierbarkeit | Nur synthetische 2D-Aufgaben / Res-Inv, Breite 64 | Offen |', '| Tiefe über 16 hinaus | Nicht getestet | Offen |', '| Reale Daten | Nicht getestet | Offen |','',r'**Primärmetrik:** \(d=\log_{10}(\mathrm{MSE}_{PreCap}/\mathrm{MSE}_{Ref})\); negativ ist besser für PreCap. Klammern: Siege / gepaarte Seeds.','', '## Leistungsprofil','','| Aufgabe | Tiefe | vs. SiLU | vs. GELU | Interpretation |','|---|---:|---|---|---|']
for t in T:
 for d in D:
  rows=[(k,m) for k,m in M.items() if k[0]==t and k[1]==d];a=eff(rows,'silu');b=eff(rows,'gelu');label='robust gegen beide' if st.median(a)<0 and st.median(b)<0 else ('nur gegen SiLU' if st.median(a)<0 else 'kein Vorteil')
  L.append(f'| {t} | {d} | {cell(rows,"silu")} | {cell(rows,"gelu")} | {label} |')
L+=['','## Aggregiert nach Tiefe','','| Tiefe | vs. SiLU | vs. GELU | Lesart |','|---:|---|---|---|']
for d in D:
 rows=[(k,m) for k,m in M.items() if k[1]==d]; a=eff(rows,'silu');b=eff(rows,'gelu');L.append(f'| {d} | {cell(rows,"silu")} | {cell(rows,"gelu")} | {"klar" if st.median(b)<-0.1 else "schwach / aufgabenabhängig"} |')
L+=['','## Stabilitätsprofil','','| Kennzahl | Gegen SiLU | Gegen GELU | Interpretation |','|---|---:|---:|---|']
for name,field in [('Aktivierungspeak: Median PreCap/Ref','hmax_peak'),('Finaler Gradient: Median PreCap/Ref','grad_final')]:
 vals=[]
 for ref in ['silu','gelu']:vals.append(st.median(m['precap'][field]/m[ref][field] for m in S.values()))
 L.append(f'| {name} | {vals[0]:.3f} | {vals[1]:.3f} | unter 1: niedrigere Extreme |')
L.append(f"| Finaler Cap-Anteil, Median | {st.median(m['precap']['cap_final'] for m in S.values()):.3f} | — | selektive Intervention |")
L+=['','## Aussagegrenzen','','- **Gesichert:** Unter diesem Trainingsprotokoll verbessert PreCap die Test-MSE robust auf `radial_ripple` und `ridge_mix`; gegen SiLU auch auf `corner_kink`.','- **Nicht gesichert:** Ein allgemeiner Vorteil gegen GELU auf knickartigen Zielstrukturen, ein wachsender Vorteil mit Tiefe, oder Übertragbarkeit auf reale Daten.','- **Wichtig:** Abnehmende Effektgröße mit Tiefe ist hier ein Befund, keine Regel über PreCap im Allgemeinen.','', '## Nächster großer Lauf','','| Achse | Eingefrorene Entscheidung vor Start | Erfolgskriterium |','|---|---|---|','| Formel | Unverändert: `precap-v9` | Keine Formeländerung |','| Architektur | Res-Inv, Breite 64; Tiefen 4, 8, 16, 32, 64 | Effektprofil über Tiefe messbar |','| Aufgaben | Bestehende 3 plus neue, vorab spezifizierte glatte/hochfrequente/knickartige 2D-Holdout-Familien | Keine Auswahl nach Ergebnis |','| Seeds | Neue, noch nicht verwendete Seed-Liste | Vollständig gepaart |','| Datenrealismus | Separater Benchmark mit festem Protokoll, nicht in die 2D-Hauptentscheidung mischen | Bericht getrennt ausweisen |','| Primärgate | Mindestens 2/3 Kernaufgaben: Median d < 0 gegen SiLU und GELU; keine höhere bad-rate | PASS/FAIL |','', '## Pflegeprotokoll','','- Für einen **identischen, vorab spezifizierten** Folgelauf: Ergebnisdatei in einen neuen Studienordner schreiben und diese Tabelle mit einem neuen, datierten Abschnitt ergänzen; den v2-Abschnitt niemals überschreiben.','- Für neue Architekturen, Daten oder Tiefen: eigene Scorecard-Sektion und getrennte PASS/FAIL-Entscheidung erstellen.','- Jede Änderung an Formel oder Parametern definiert eine neue Aktivierungsvariante, nicht eine Aktualisierung von PreCap-v9.']
open('docs/Evidenz_Bewertungstabelle.md','w').write('\n'.join(L))
print('wrote docs/Evidenz_Bewertungstabelle.md')
