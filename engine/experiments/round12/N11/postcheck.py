import runpy
from pathlib import Path
folder=Path(__file__).resolve().parent
for name in ('certificates.py','report.py'):runpy.run_path(str(folder/name),run_name='__main__')
