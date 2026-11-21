"""Source-preserving tuple-arity bridge after the zero-row export failure."""
import importlib.util
from pathlib import Path
import n9lib as L
original_encode=L.encode;original_save=L.save
def bridge(g,obs):
    x,mask,prop,public=original_encode(g,obs)
    return x,mask,prop,None,public # unused compatibility slot; no value/input change
def preserve_selection(path,value):
    path=Path(path)
    if path.name=='selection.json' and path.exists():
        old=L.json.loads(path.read_text());assert {k:v for k,v in old.items() if k!='utc'}=={k:v for k,v in value.items() if k!='utc'};return
    original_save(path,value)
def main():
    p=Path(__file__);adapter={str(p):L.sha(p)};receipt=L.O/'api_adapter.json'
    if receipt.exists():assert L.json.loads(receipt.read_text())['sources']==adapter
    else:original_save(receipt,{'utc':L.stamp(),'sources':adapter,'original_source_freeze_sha256':L.sha(L.O/'sources.json'),'scope':'Only maps4encoderreturnvalues to5withunusedNone; preserves prior selectionIDs/UTC; no original scientific source or target/count/seed/model change. Zero rows/fits existed at failure.'})
    # Public-observation-only fixture, no hidden labels/fits.
    g=L.game(L.TAGS[0]);folder=L.R/'N2/results'/L.TAGS[0]
    with L.np.load(folder/'posterior_supports.npz') as d:key=d['keys'][0]
    obs=L.observation(g,key);a=original_encode(g,obs);b=bridge(g,obs)
    for left,right in zip(a[:3],b[:3]):L.np.testing.assert_array_equal(left,right)
    assert a[3].keys()==b[4].keys() and b[3] is None
    L.encode=bridge;L.save=preserve_selection
    spec=importlib.util.spec_from_file_location('n9_frozen_science',L.H/'run.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);oldguard=m.guard
    def guard(sources):
        oldguard(sources);assert all(L.sha(Path(p))==v for p,v in adapter.items()),'Adapter drift'
    m.guard=guard
    L.log(L.H,'Encoderarity adapter public fixture passed; originalsource/selection guards active; unchanged science/count resume.')
    m.main()
if __name__=='__main__':main()
