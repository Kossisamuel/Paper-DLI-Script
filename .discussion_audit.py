import json
from pathlib import Path

root = Path.cwd()
for parent in [root, *root.parents]:
    instructions = parent / 'AGENTS.md'
    if instructions.exists():
        print(instructions)
        print(instructions.read_text(encoding='utf-8'))
print('REQUEST')
print(Path(r'C:\Users\Benoit\.codex\attachments\c9192baf-d748-4bc0-a3b4-bb35ce92ec55\pasted-text.txt').read_text(encoding='utf-8'))
notebook = json.loads(Path('discussiondli.ipynb').read_text(encoding='utf-8'))
print('CELLS', len(notebook['cells']))
for index, cell in enumerate(notebook['cells']):
    print('\nCELL', index, cell['cell_type'])
    print(''.join(cell['source']))
