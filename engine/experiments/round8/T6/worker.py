"""Run the focused bidder checks before the frozen-candidate study."""
from pathlib import Path
import os,runpy
HERE=Path(__file__).resolve().parent
os.chdir(HERE.parents[2])
import pytest
code=pytest.main(['-q','-p','no:cacheprovider','tests/test_round8_human_v2.py'])
with (HERE/'STATUS.txt').open('a',encoding='utf-8') as f:
    f.write(f'Focused V2 test exit code {code}; command python -m pytest -q -p no:cacheprovider tests/test_round8_human_v2.py from engine.\n')
if code:raise SystemExit(code)
runpy.run_path(str(HERE/'run.py'),run_name='__main__')
