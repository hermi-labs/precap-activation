import json,subprocess,sys
from pathlib import Path
r=json.loads(subprocess.check_output([sys.executable,'run_v12_diagnosis.py','--integrity']))
print({'full_runs':f"{r['full_runs']}/{r['expected_full_runs']}",'checkpoint_rows':f"{r['checkpoint_rows']}/{r['expected_checkpoint_rows']}",'done':Path('done.txt').exists()})
