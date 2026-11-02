# C6 starten (Windows/Linux/macOS, ohne screen): Datenpruefung (SHA-256 wie data/cifar10/SHA256.txt), Einfrieren, Selbsttest,
# Aequivalenztest gegen die MLX-Referenz auf dem Zielgeraet, Auswertungstest, Rauchtest, dann N Worker als losgeloeste Prozesse mit Logs.
# START NUR NACH AARONS FREIGABE. Mit Claude geschrieben.
# Aufruf: python c6_launch.py [--device cuda] [--workers 2] [--nur-tests]
import argparse, hashlib, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path
H = Path(__file__).resolve().parent; PY = sys.executable
DATA = Path(os.environ.get('PRECAP_DATA', str(Path.home() / 'precap_data')))
SHA = {'data_batch_1.bin': 'cee916563c9f80d84e3cc88e17fdc0941787f1244f00a67874d45b261883ada5', 'data_batch_2.bin': 'a591ca11fa1708a91ee40f54b3da4784ccd871ecf2137de63f51ada8b3fa57ed',
       'data_batch_3.bin': 'bbe8596564c0f86427f876058170b84dac6670ddf06d79402899d93ceea26f67', 'data_batch_4.bin': '014e562d6e23c72197cc727519169a60359f5eccd8945ad5a09d710285ff4e48',
       'data_batch_5.bin': '755304fc0b379caeae8c14f0dac912fbc7d6cd469eb67a1029a08a39453a9add', 'test_batch.bin': '8e2eb146ae340b09e24670f29cabc6326dba54da8789dab6768acf480273f65b'}
FREEZE = ['c6_run.py', 'c6_analyze.py', 'c6_launch.py', 'C6_Praeregistrierung.md', 'test/test_c6_aequivalenz.py', 'test/test_c6_analyse.py', 'test/mlx_ref.npz', 'test/mlx_ref2.npz',
          'mlx_vergleich/SHA256.txt']
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def log(msg):
    (H / 'logs').mkdir(exist_ok=True)
    with open(H / 'logs' / 'launch.txt', 'a', encoding='utf-8') as f: f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")
    print(msg, flush=True)
def stop(msg): log('ABBRUCH: ' + msg); sys.exit(1)
def step(name, args, must, out):
    env = dict(os.environ, PRECAP_DATA=str(DATA), PYTHONIOENCODING='utf-8')
    r = subprocess.run([PY] + args, cwd=H, capture_output=True, text=True, env=env); (H / 'meta').mkdir(exist_ok=True)
    (H / 'meta' / out).write_text(r.stdout + '\n--- stderr ---\n' + r.stderr, encoding='utf-8')
    if r.returncode != 0 or must not in r.stdout: stop(f'{name} nicht ok (siehe meta/{out})')
    log(f'{name} ok')
def main():
    p = argparse.ArgumentParser(); p.add_argument('--device', default='cuda'); p.add_argument('--workers', type=int, default=2); p.add_argument('--nur-tests', action='store_true'); a = p.parse_args()
    if (H / 'workers').exists(): stop('Ergebnisordner workers/ existiert schon')
    for f, h in SHA.items():
        q = DATA / 'cifar10' / 'cifar-10-batches-bin' / f
        if not q.exists() or sha(q) != h: stop(f'Datei {q} fehlt oder SHA-256 weicht ab')
    log('Daten ok (6 Dateien, SHA-256 wie auf dem Mac)')
    import torch
    info = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda, cuda_ok=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, platform=sys.platform)
    log('Umgebung ' + json.dumps(info))
    if a.device == 'cuda' and not info['cuda_ok']: stop('CUDA nicht verfuegbar')
    (H / 'meta').mkdir(exist_ok=True)
    with open(H / 'meta' / 'frozen_sha256.txt', 'w', encoding='utf-8') as f:
        for x in FREEZE: f.write(f'{sha(H / x)}  {x}\n')
        for x in SHA: f.write(f'{SHA[x]}  DATA/cifar10/cifar-10-batches-bin/{x}\n')
    (H / 'meta' / 'frozen_at.txt').write_text(datetime.now().astimezone().isoformat(timespec='seconds'), encoding='utf-8'); log('eingefroren')
    step('Selbsttest', ['c6_run.py', '--selfcheck', '--device', a.device], '"ok": true', 'selfcheck.json')
    step('Aequivalenztest gegen MLX', ['test/test_c6_aequivalenz.py', a.device], 'AEQUIVALENZ OK', f'aequivalenz_{a.device}.txt')
    step('Auswertungstest', ['test/test_c6_analyse.py'], 'AUSWERTUNG OK', 'auswertungstest.txt')
    step('Rauchtest', ['c6_run.py', '--smoke', '--device', a.device], 'smoke c6_tat', 'rauchtest.txt')
    if a.nur_tests: log('nur Tests, kein Start'); return
    flags = 0
    if sys.platform == 'win32': flags = 0x00000008 | 0x00000200 | 0x01000000   # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB
    env = dict(os.environ, PRECAP_DATA=str(DATA), PYTHONIOENCODING='utf-8', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for w in range(a.workers):
        lf = open(H / 'logs' / f'w{w}.log', 'a', encoding='utf-8')
        args = [PY, 'c6_run.py', '--worker', str(w), '--workers', str(a.workers), '--device', a.device]
        try: pr = subprocess.Popen(args, cwd=H, stdout=lf, stderr=subprocess.STDOUT, env=env, creationflags=flags, start_new_session=(sys.platform != 'win32'))
        except OSError:   # Ausbrechen aus dem Job der SSH-Sitzung nicht erlaubt: ohne Breakaway starten und warnen
            pr = subprocess.Popen(args, cwd=H, stdout=lf, stderr=subprocess.STDOUT, env=env, creationflags=flags & ~0x01000000); log('WARNUNG: Breakaway nicht erlaubt, SSH-Sitzung offen lassen oder per Aufgabenplanung starten')
        log(f'Worker {w} gestartet, PID {pr.pid}')
    log(f'gestartet, {a.workers} Worker, 60 Laeufe')
if __name__ == '__main__': main()
