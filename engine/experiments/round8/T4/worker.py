"""Optional freed compute slot; the task lock prevents duplicate simulation."""
from pathlib import Path
import runpy,os
HERE=Path(__file__).resolve().parent
os.chdir(HERE.parents[2])
import pytest
code=pytest.main(['-q','-p','no:cacheprovider','tests/test_round8_lead.py'])
with (HERE/'STATUS.txt').open('a',encoding='utf-8') as f:
    f.write(f'Focused method tests before external worker: exit code {code}; command python -m pytest -q -p no:cacheprovider tests/test_round8_lead.py from engine.\n')
if code:
    raise SystemExit(code)
runpy.run_path(str(HERE/'run.py'),run_name='__main__')
if not (HERE/'worker.lock').exists():
    runpy.run_path(str(HERE/'analyze.py'),run_name='__main__')
