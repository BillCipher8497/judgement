"""Bounded-token seed collision audit against pre-round tracked source/text."""
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import save

BASES = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 2000, 2010, 2020]
paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
hits = []
for rel in paths:
    p = ROOT/rel
    if not p.is_file() or p.suffix.lower() not in ('.py', '.md', '.txt', '.json', '.sh', '.ps1', '.csv', '.yaml', '.yml'): continue
    if p.stat().st_size > 20_000_000: continue
    try: content = p.read_text(encoding='utf-8-sig')
    except UnicodeError: continue
    for b in BASES:
        # Numeric full seed or conventional M spelling. A fraction substring
        # must not be treated as a registered seed without inspecting context.
        pattern = rf'(?<![\w.])(?:{b}000000|{b}\s*M)(?![\w.])'
        for match in re.finditer(pattern, content):
            hits.append({'base_M': b, 'path': rel, 'line': content.count('\n', 0, match.start())+1,
                         'context': content[max(0, match.start()-70):match.end()+70]})
import numpy, scipy, pytest
save(Path(__file__).parent/'results/inventory.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
     'python': sys.version, 'executable': sys.executable, 'numpy': numpy.__version__,
     'scipy': scipy.__version__, 'pytest': pytest.__version__, 'reserved_bases_M': BASES,
     'tracked_text_matches': hits, 'searched_tracked_paths': len(paths)-1,
     'limitations': 'Tracked text files <=20MB; inspect any matches; excludes third-party PDFs and binary arrays.'})
print(json.dumps({'matches': hits, 'paths': len(paths)-1}))
