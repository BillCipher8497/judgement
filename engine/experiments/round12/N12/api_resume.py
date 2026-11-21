"""Source-preserving JSON scalar adapter after first output serialization failed."""
from lib12 import *
import runpy
original=save
def native(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [native(v) for v in x]
    return x
def wrapped(path,value):return original(path,native(value))
def main():
    import lib12
    source=O/'sources.json';binding={'source_sha256':sha(source),'adapter_sha256':sha(Path(__file__)),'scope':'ONLY numpy scalar to identical native JSON number; no data/payoff/model/selection/seed/decoder change'};p=O/'json_adapter.json'
    if p.exists():assert json.loads(p.read_text())==binding
    else:original(p,binding)
    assert native({'v':np.int64(3),'b':np.bool_(True)})=={'v':3,'b':True};lib12.save=wrapped
    runpy.run_path(str(H/'run.py'),run_name='__main__')
if __name__=='__main__':main()
