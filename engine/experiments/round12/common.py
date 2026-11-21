"""Round-local checkpoint helpers, with bounded Windows sharing retries."""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''): h.update(b)
    return h.hexdigest()


def replace(temp, path):
    for attempt in range(40):
        try: os.replace(temp, path); return
        except PermissionError:
            if attempt == 39: raise
            time.sleep(.1)


def save(path, data):
    path = Path(path); path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8'); replace(temp, path)


def stamp(): return datetime.now(timezone.utc).isoformat()


def log(folder, message):
    with (Path(folder)/'STATUS.txt').open('a', encoding='utf-8') as f: f.write(stamp()+' '+message+'\n')
    print(message, flush=True)
