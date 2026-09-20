import ast
import json
from pathlib import Path

path = Path('discussiondli.ipynb')
notebook = json.loads(path.read_text(encoding='utf-8'))
replacements = {
    'import json\nimport sys': 'import json\nimport os\nimport sys',
    'SMOKE_TEST = False\nN_JOBS = -1': "SMOKE_TEST = os.environ.get('MI_QUICK_RUN', '0') == '1'\nN_JOBS = int(os.environ.get('MI_N_JOBS', '1'))",
    "PCA(n_components=components, random_state=RANDOM_STATE)": "PCA(\n            n_components=components, svd_solver='full', random_state=RANDOM_STATE,\n        )",
    '`SMOKE_TEST=True` runs': '`MI_QUICK_RUN=1` sets `SMOKE_TEST=True` and runs',
    'Set it to `False` for the original full grids': 'Leave it unset or set it to `0` for the original full grids',
    'patients, both scenarios, and the same preprocessing. It checks execution only.': 'patients, both scenarios, and the same preprocessing. It checks execution only.\n`MI_N_JOBS` controls parallel grid-search jobs and defaults to `1`.',
}
for cell in notebook['cells']:
    source = ''.join(cell['source'])
    for old, new in replacements.items():
        source = source.replace(old, new)
    cell['source'] = source.splitlines(keepends=True)
    if cell['cell_type'] == 'code':
        ast.parse(source)
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
try:
    import nbformat
    nbformat.validate(nbformat.read(path, as_version=4))
    print('Notebook schema and Python syntax validation passed.')
except ImportError:
    print('Python syntax passed; nbformat is not installed in this interpreter.')
