"""Summarize only matched complete pairs; never count unfinished games."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
rows = {}
ladder=np.array(list(range(13,0,-1))+list(range(2,14)))
for p in (HERE/'results').glob('*_*.json'):
    d = json.loads(p.read_text())
    if 'g' in d:
        assert d['seed_base']==1000000000 and d['seat']==d['g']%4
        b,t,s=np.asarray(d['bids']),np.asarray(d['tricks']),np.asarray(d['scores'])
        assert b.shape==t.shape==s.shape==(25,4)
        np.testing.assert_array_equal(t.sum(1),ladder)
        assert np.all(b.sum(1)!=ladder)
        np.testing.assert_array_equal(s,np.where(b==t,10+b,0))
        np.testing.assert_array_equal(s.sum(0),d['totals'])
        rows[d['g'],d['arm']] = d
out = {}
for arm in ['sanity','rb6','rb7']:
    games = sorted(g for g,a in rows if a == arm and (g,'ref') in rows)
    if len(games)<2:
        continue
    ds, ws = [], []
    for g in games:
        r,a = rows[g,'ref'],rows[g,arm]
        seat = g%4
        ds.append(a['totals'][seat]-r['totals'][seat])
        ws.append(int(a['totals'][seat] == max(a['totals']))-int(r['totals'][seat] == max(r['totals'])))
    def ci(v):
        return [float(np.mean(v)), float(1.96*np.std(v,ddof=1)/np.sqrt(len(v)))]
    out[arm] = {'n':len(games),'games':games,'score_mean_half95':ci(ds),'ties_win_mean_half95':ci(ws),
        'runtime_seconds_mean_half95':ci([rows[g,arm]['seconds'] for g in games]),
        'reference_runtime_seconds_mean_half95':ci([rows[g,'ref']['seconds'] for g in games]),
        'identical_total_vectors':sum(rows[g,arm]['totals']==rows[g,'ref']['totals'] for g in games),
        'identical_bid_vectors':sum(rows[g,arm]['bids']==rows[g,'ref']['bids'] for g in games)}
games=sorted(g for g,a in rows if a=='rb7' and (g,'rb6') in rows)
if len(games)>=2:
    score=[];wins=[]
    for g in games:
        a,b=rows[g,'rb7'],rows[g,'rb6'];seat=g%4
        score.append(a['totals'][seat]-b['totals'][seat])
        wins.append(int(a['totals'][seat]==max(a['totals']))-int(b['totals'][seat]==max(b['totals'])))
    out['rb7_minus_rb6']={'n':len(games),'games':games,'score_mean_half95':ci(score),
        'ties_win_mean_half95':ci(wins),'interpretation':'Predeclared depth control: same RB sampler settings and K32, c7 versus c6.'}
path = HERE/'results'/'summary.json'
path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:{name:value for name,value in v.items() if name!='games'} for k,v in out.items()},indent=2))
