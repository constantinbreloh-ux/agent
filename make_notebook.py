"""Builds gga_autopilot.ipynb from the repo files: python make_notebook.py, then import it in Kaggle."""
import json
from pathlib import Path

here = Path(__file__).parent
cells = [
    ('markdown', '# GGA AutoPILOT: Gemma 4 31B with macros and live controllers\n'
                 'Inputs: the competition data and gemma-4-31b-it-qat-w4a16-ct. Accelerator: RTX Pro 6000. '
                 'Internet: off. Built by `make_notebook.py`, do not edit here.'),
    ('code', (here / 'notebook/setup.py').read_text()),
    ('code', '%%writefile /kaggle/working/autopilot.py\n' + (here / 'autopilot.py').read_text()),
    ('code', (here / 'gemma.py').read_text()),
    ('code', (here / 'notebook/play.py').read_text()),
    ('code', (here / 'notebook/submit.py').read_text()),
]
notebook = {
    'cells': [{'cell_type': kind, 'metadata': {}, 'source': src.splitlines(keepends=True),
               **({'outputs': [], 'execution_count': None} if kind == 'code' else {})} for kind, src in cells],
    'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}},
    'nbformat': 4, 'nbformat_minor': 5,
}
(here / 'gga_autopilot.ipynb').write_text(json.dumps(notebook, indent=1))
print('wrote gga_autopilot.ipynb')
