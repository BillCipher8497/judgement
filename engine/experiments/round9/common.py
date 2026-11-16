"""Round-9 checkpoint/board helpers; workers never create child processes."""
import json,os,threading,time
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
DEADLINE=datetime(2026,11,16,22,30,tzinfo=timezone.utc).timestamp()
def log(task,message):
    until=time.time()+30
    while True:
        try:
            with (HERE/task/'STATUS.txt').open('a',encoding='utf-8') as f:
                f.write(datetime.now(timezone.utc).isoformat()+' '+message+'\n')
            return
        except PermissionError:
            if time.time()>until:raise
            time.sleep(.2)
def replace(source,target):
    until=time.time()+30
    while True:
        try:source.replace(target);return
        except PermissionError:
            if time.time()>until:raise
            time.sleep(.2)
def save(path,value):
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    payload=json.dumps(value,indent=2,allow_nan=False);until=time.time()+30
    while True:
        try:tmp.write_text(payload,encoding='utf-8');break
        except PermissionError:
            if time.time()>until:raise
            time.sleep(.2)
    replace(tmp,path)
def status(task,state,note):
    path=HERE/'BOARD.md';lines=path.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| '+task+' |'):
            cells=line.split('|');cells[-3]=' '+state+' ';cells[-2]=' '+note+' ';lines[i]='|'.join(cells)
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
def timer(limit=DEADLINE):
    if (HERE/'PAUSED').exists():raise SystemExit('Round 9 PAUSED marker present; read its current hold/resume instructions before launching.')
    stop=min(limit,DEADLINE);remaining=stop-time.time()
    if remaining<=0:raise SystemExit('Round-9 compute deadline reached')
    class AbsoluteStop:
        def __init__(self):
            self.done=threading.Event()
            self.thread=threading.Thread(target=self.watch,daemon=True)
            self.thread.start()
        def watch(self):
            while not self.done.is_set():
                left=stop-time.time()
                if left<=0:os._exit(0)
                self.done.wait(min(.25,left))
        def cancel(self):self.done.set()
    return AbsoluteStop()
