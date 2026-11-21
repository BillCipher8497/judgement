"""Synthetic gradients/activation and immutable old-context private rollout fixtures."""
from n10lib import *
from reference import rollout
def main():
    registry=Registry();registry.add([FITBASE,DOMAIN,99]);rng=np.random.default_rng([FITBASE,DOMAIN,99]);x=rng.normal(0,.1,(4,683));base=np.array([[.25,0],[0,.25]]*2);p=initialize([683,64,32,2],[FITBASE,DOMAIN,99]);errors=[]
    for arm in ARMS:
        target=rng.uniform(.1,.9,(4,2))
        if arm=='hard':target/=target.sum(1,keepdims=True)
        _,gg=grad(x,base,target,np.ones(4),p,arm)
        for k in range(len(p)):
            for index in rng.choice(p[k].size,min(3,p[k].size),replace=False):
                j=np.unravel_index(index,p[k].shape);old=p[k][j];eps=1e-5;p[k][j]=old+eps;plus=grad(x,base,target,np.ones(4),p,arm)[0];p[k][j]=old-eps;minus=grad(x,base,target,np.ones(4),p,arm)[0];p[k][j]=old;err=abs((plus-minus)/(2*eps)-gg[k][j]);assert err<3e-6;errors.append(err)
    # Prove expressivity/nonidentity gate isn't structurally impossible, without
    # using these artificial weights in any fitted policy or game.
    p=[np.zeros((683,64)),np.zeros(64),np.zeros((64,32)),np.zeros(32),np.zeros((32,2)),np.zeros(2)];p[0][681,0]=1;p[0][682,1]=1;p[2][0,0]=1;p[2][1,1]=1;p[4][0,1]=2;p[4][1,0]=2
    fx=np.zeros((2,683));fx[0,681]=fx[1,682]=1;fb=np.array([[.25,0],[0,.25]]);assert np.array_equal(scores({'x':fx,'base':fb},p,1).argmax(1),[1,0]);p[-2][:]=0;assert np.array_equal(scores({'x':fx,'base':fb},p,1).argmax(1),[0,1])
    oldfolder=R/'N3/results/data/pilot';path=sorted(oldfolder.glob('*.npz'))[0];data=load(path);meta=json.loads(path.with_suffix('.json').read_text());n=0
    for k in range(8):
        o=meta['observations'][k];info=OBS.restore(o);hero=o['seat'];assert not hasattr(info,'_st')
        bcard=int(data['cards'][k,data['base'][k].argmax()]);xx,bb,cc=OBS.encode(info,sum(1<<int(c) for c in data['cards'][k]),bcard);np.testing.assert_array_equal(xx,data['x'][k])
        for w in range(2):
            hidden={p:int(data['worlds'][k,w,p]) for p in range(4) if p!=hero};address=[FITBASE,DOMAIN,99,k,w];registry.add(address)
            for card in cc:
                primary=private_rollout(info,hidden,card,hero,address);fresh=rollout(o,hidden,card,hero,address);assert primary==fresh;n+=1
    save(H/'self_check.json',{'utc':stamp(),'n_gradient_entries':len(errors),'max_gradient_error':max(errors),'activation_counterexample_and_zero_head_identity':True,'n_private_rollout_transition_pairs':n,'immutable_old_observable_features_match':True,'no_fullstate_capability':True,'seed_receipt':registry.receipt(),'source_sha256':sha(Path(__file__)),'scope':'Synthetic/old immutable fixtures only; no newN10data/fits/games'});print('N10synthetic/private fixtures passed',n,max(errors),flush=True)
if __name__=='__main__':main()
