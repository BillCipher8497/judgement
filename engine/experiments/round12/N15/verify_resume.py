"""Source-preserving import-path adapter for the frozen separate reader."""
import sys,runpy,json
from pathlib import Path
TASK=Path(__file__).resolve().parent;sys.path.insert(0,str(TASK.parent/'N13'))
import lib13
def main():
    out=TASK/'results';binding={'source_freeze_sha256':lib13.sha(out/'sources.json'),'adapter_sha256':lib13.sha(Path(__file__)),'original_reader_sha256':lib13.sha(TASK/'verify.py'),'scope':'ONLY preload unchanged N13 rational helper so its import by file path resolves; no arithmetic/model/data/selection/seed edit'};p=out/'verification_adapter.json'
    if p.exists():assert json.loads(p.read_text())==binding
    else:lib13.save(p,binding)
    runpy.run_path(str(TASK/'verify.py'),run_name='__main__')
if __name__=='__main__':main()
