"""Second compute slot: resumable T3, then T4 and time-boxed T5. No subprocesses."""
from pathlib import Path
import runpy, datetime, time, sys
HERE=Path(__file__).resolve().parent

def update(task,status,note):
    board=HERE/'BOARD.md';lines=board.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| '+task+' |'):
            cells=line.split('|');cells[-3]=' '+status+' ';cells[-2]=' '+note+' ';lines[i]='|'.join(cells)
    board.write_text('\n'.join(lines)+'\n',encoding='utf-8')

for task in (sys.argv[1:] or ['T3','T4','T5']):
    if task not in ['T3','T4','T5']:
        raise ValueError(task)
    # Claude may request fixes while the previous task computes. Stop scheduling
    # new tasks in that case; root assistant handles requested changes.
    if any(line.startswith('| T') and len(line.split('|'))>5
           and line.split('|')[-3].strip().lower()=='changes-requested'
           for line in (HERE/'BOARD.md').read_text(encoding='utf-8').splitlines()):
        break
    state=next(line.split('|')[-3].strip() for line in (HERE/'BOARD.md').read_text(encoding='utf-8').splitlines() if line.startswith('| '+task+' |'))
    if state in ['validated','ready-for-review']:
        continue
    if task in ['T4','T5'] and (HERE/task/'worker.lock').exists():
        continue  # An independently allocated slot owns the task; preserve it.
    update(task,'running','Second compute slot active; raw outcomes checkpointed; hard 07:30 ET deadline. See task STATUS for scope and replication seeds.')
    runpy.run_path(str(HERE/task/'run.py'),run_name='__main__')
    if task=='T3':
        import pytest, os
        os.chdir(HERE.parents[1])
        code=pytest.main(['-q','-p','no:cacheprovider'])
        with (HERE/task/'STATUS.txt').open('a',encoding='utf-8') as f:
            f.write(f'Integration test suite exit code: {code}.\n')
        if code:break
    if (HERE/task/'analyze.py').exists():
        # Local import `run` in T4 analysis must resolve within this task.
        import sys
        sys.path.insert(0,str(HERE/task));sys.modules.pop('run',None)
        runpy.run_path(str(HERE/task/'analyze.py'),run_name='__main__')
    with (HERE/task/'STATUS.txt').open('a',encoding='utf-8') as f:
        f.write('COMPUTE AND ANALYSIS FINISHED. Assistant must append review handoff with file list, exact commands, seeds, n, CIs and limitations before ready-for-review.\n')
    current=next(line.split('|')[-3].strip() for line in (HERE/'BOARD.md').read_text(encoding='utf-8').splitlines() if line.startswith('| '+task+' |'))
    if current not in ['ready-for-review','validated']:
        update(task,'running','Compute finished; awaiting assistant result checks and complete review handoff before ready-for-review.')
