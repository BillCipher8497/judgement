"""N2 trajectory exporter prototype. No CLI starts scientific work.
Every full fixed-policy play infoset is retained, including forced decisions.
"""
import json,hashlib,time
from itertools import combinations
from pathlib import Path
import numpy as np
from bid_premise import assert_matched
KEY_WIDTH=20

def deals(deck,n,hands=()):
    if len(hands)==3:
        yield hands+(tuple(deck),);return
    for hand in combinations(deck,n):
        taken=set(hand)
        yield from deals(tuple(c for c in deck if c not in taken),n,hands+(tuple(hand),))

def key(seat,ownhand,hist):
    hmask=sum(1<<c for c in ownhand);h=bytes(hist)
    packed=bytes([seat,len(h)])+hmask.to_bytes(2,'little')+h
    assert len(packed)<=KEY_WIDTH
    return packed.ljust(KEY_WIDTH,b'\0')

def decode(raw):
    raw=bytes(raw).ljust(KEY_WIDTH,b'\0');seat,L=raw[:2]
    mask=int.from_bytes(raw[2:4],'little');hist=tuple(raw[4:4+L])
    assert 4<=L<=15
    return seat,tuple(c for c in range(12) if mask>>c&1),hist

def trajectory(g,hands,choose_bid,choose_card):
    bids=[]
    for p in range(4):
        b=int(choose_bid(g,p,hands[p],tuple(bids)));assert b in g.legal_bids(bids);bids.append(b)
    matched=assert_matched(g,hands,tuple(bids))
    hist=tuple(bids)
    for event in range(4*g.n):
        assert matched # Every exported key/world satisfies R subset F_bid.
        left,_,trick,_,_,actor=g.replay(hands,hist);legal=g.legal_cards(left[actor],trick)
        assert legal
        c=legal[0] if len(legal)==1 else int(choose_card(g,actor,hands[actor],hist));assert c in legal
        unseen=set(g.deck)-set(hands[actor])-set(hist[4:]);holders=[(actor+k)%4 for k in (1,2,3)];labels=0
        for card in unseen:
            j=next(j for j,p in enumerate(holders) if card in hands[p]);labels|=j<<(2*card)
        # True hand labels never enter the private key or choose_card inputs.
        yield key(actor,hands[actor],hist),labels,c
        hist+=(c,)
    assert len(hist)==4+4*g.n

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_npz(path,**arrays):
    path=Path(path);temp=path.with_suffix('.tmp')
    if temp.exists():temp.rename(temp.with_name(temp.name+'.orphan_'+str(time.time_ns())))
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(path)

def write_chunk(g,folder,start,hands_rows,choose_bid,choose_card,freeze,deadline):
    assert time.time()<deadline;folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);p=folder/f'{start:06d}.npz';receipt=p.with_suffix('.json')
    if receipt.exists():
        r=json.loads(receipt.read_text());assert r['source_freeze_sha256']==freeze and r['raw_sha256']==digest(p)
        assert r['start_deal']==start and r['n_deals']==len(hands_rows)
        assert r['condition']=={'N':g.N,'S':g.S,'R':g.R,'n':g.n,'trump':g.trump}
        return r
    if p.exists():p.rename(p.with_name(p.name+'.orphan_'+str(time.time_ns())))
    keys=[];labels=[];chosen=[];dealids=[];tick=time.perf_counter()
    for k,hands in enumerate(hands_rows):
        assert time.time()<deadline
        for private,y,c in trajectory(g,hands,choose_bid,choose_card):keys.append(private);labels.append(y);chosen.append(c);dealids.append(start+k)
    assert len(keys)==len(hands_rows)*4*g.n
    atomic_npz(p,keys=np.array(keys,dtype=f'S{KEY_WIDTH}'),holder_bits=np.array(labels,np.uint32),chosen_card=np.array(chosen,np.uint8),deal_index=np.array(dealids,np.uint32))
    result=dict(condition={'N':g.N,'S':g.S,'R':g.R,'n':g.n,'trump':g.trump},start_deal=start,n_deals=len(hands_rows),n_records=len(keys),source_freeze_sha256=freeze,raw_sha256=digest(p),wall_seconds=time.perf_counter()-tick)
    temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(receipt);return result

def merge(g,folder,n_deals,deadline):
    """Complete fixed-profile posterior supports, without M1root selection."""
    folder=Path(folder);paths=sorted(folder.glob('*.npz'));data=[];expected=0;raw_hashes={};freezes=set();condition={'N':g.N,'S':g.S,'R':g.R,'n':g.n,'trump':g.trump}
    existing=folder.parent/'posterior_manifest.json'
    if existing.exists():
        saved=json.loads(existing.read_text());assert saved['condition']==condition and saved['n_exact_chance_deals']==n_deals
        assert saved['raw_hashes']=={p.name:digest(p) for p in paths}
        assert digest(folder.parent/'posterior_supports.npz')==saved['posterior_sha256']
        return saved
    for p in paths:
        assert time.time()<deadline
        r=json.loads(p.with_suffix('.json').read_text());assert r['condition']==condition and r['start_deal']==expected and digest(p)==r['raw_sha256']
        a=dict(np.load(p));assert len(a['keys'])==r['n_deals']*4*g.n
        np.testing.assert_array_equal(a['deal_index'],np.repeat(np.arange(expected,expected+r['n_deals']),4*g.n))
        expected+=r['n_deals'];raw_hashes[p.name]=r['raw_sha256'];freezes.add(r['source_freeze_sha256']);data.append(a)
    assert expected==n_deals and len(freezes)==1
    keys=np.concatenate([a['keys'] for a in data]);dealid=np.concatenate([a['deal_index'] for a in data]);labels=np.concatenate([a['holder_bits'] for a in data]);del data
    unique,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True);del keys
    order=np.argsort(inverse,kind='stable');support=dealid[order];offset=np.cumsum(np.r_[0,counts]);del dealid,order
    for a,b in zip(offset[:-1],offset[1:]):assert np.all(support[a+1:b]>support[a:b-1]),'One deal reached one identical play key twice'
    unknown=[]
    for raw in unique:
        _,hand,hist=decode(raw);seen=sum(1<<c for c in set(hand)|set(hist[4:]));unknown.append(((1<<(g.S*g.R))-1)^seen)
    unknown=np.array(unknown,np.uint16);holder=np.zeros((len(unique),g.S*g.R,3),np.uint32)
    for c in g.deck:
        assert time.time()<deadline
        valid=((unknown[inverse]>>c)&1).astype(bool);where=inverse[valid];y=((labels[valid]>>(2*c))&3).astype(np.intp);assert np.all(y<3)
        np.add.at(holder[:,c,:],(where,y),1)
        assert np.all(holder[:,c,:].sum(1)==counts*((unknown>>c)&1))
    target=folder.parent/'posterior_supports.npz'
    if target.exists():target.rename(target.with_name(target.name+'.orphan_'+str(time.time_ns())))
    atomic_npz(target,keys=unique,reach_count=counts.astype(np.uint32),support_offset=offset.astype(np.uint64),support_deal_index=support,holder_counts=holder,unseen_mask=unknown)
    result={'condition':condition,'source_freeze_sha256':next(iter(freezes)),'n_exact_chance_deals':n_deals,'n_recorded_decision_worlds':int(counts.sum()),'n_private_play_infosets':len(unique),'raw_hashes':raw_hashes,'posterior_sha256':digest(folder.parent/'posterior_supports.npz'),
        'definition':'Uniform complete chance deals reaching each full private/public play key under one fixed deterministic profile; no hidden root-stage flag; all forced nodes retained.'}
    (folder.parent/'posterior_manifest.json').write_text(json.dumps(result,indent=2));return result
