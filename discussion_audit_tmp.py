import json
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
n = json.loads(Path('discussiondli.ipynb').read_text(encoding='utf-8'))
for i, c in enumerate(n['cells']):
    if 7 <= i <= 26:
        print('\nCELL', i, c['cell_type'])
        print(''.join(c['source']))
