"""Pre-fit gradient and proper-normalization fixtures on synthetic observations."""
from n9lib import *
def main():
    rng=np.random.default_rng([BASE,DOMAIN,99]);batch=[]
    for U,caps in ((3,(1,1,1)),(4,(2,1,1))):
        allowed=np.ones((U,3),bool);assigned=[]
        for hs in allocations(caps,allowed):
            own=np.empty(U,np.uint8)
            for j,h in enumerate(hs):own[list(h)]=j
            assigned.append(own)
        F=np.array(assigned);post=np.array([0,len(F)-1]);target=np.zeros((U,4))
        for owners in F[post]:target[np.arange(U),owners]+=.5
        mask=np.c_[allowed,np.zeros(U,bool)];prop=np.tile(np.r_[caps,0]/U,(U,1));x=rng.normal(0,.2,(U,4,245))
        batch.append(SimpleNamespace(x=x,mask=mask,prop=prop,target=target,F=F,posterior=post,count=2))
    errors=[]
    for arch in (0,1):
        p=initialize(arch,[BASE,DOMAIN,99]);p[-2][:]=rng.normal(0,.02,p[-2].shape)
        for objective in LOSSES:
            value,grad=loss_gradient(batch,p,objective,True,2)
            for k in range(len(p)):
                for index in rng.choice(p[k].size,min(4,p[k].size),replace=False):
                    ix=np.unravel_index(index,p[k].shape);old=p[k][ix];eps=1e-5;p[k][ix]=old+eps;plus=loss_gradient(batch,p,objective,True,2)[0];p[k][ix]=old-eps;minus=loss_gradient(batch,p,objective,True,2)[0];p[k][ix]=old
                    err=abs((plus-minus)/(2*eps)-grad[k][ix]);assert err<3e-6;errors.append(err)
            for row in batch:
                z=score(row,p,True);kl,prob,lz=joint(z,row);assert abs(prob.sum()-1)<1e-14
                direct=-sum(np.log(prob[j]) for j in row.posterior)/row.count-np.log(row.count);assert abs(kl-direct)<1e-12
                assert abs(joint(z+np.arange(len(z))[:,None],row)[0]-kl)<1e-12
    save(H/'self_check.json',{'utc':stamp(),'n_finite_difference_entries':len(errors),'max_error':max(errors),'joint_direct_probability_and_logshift_invariance':True,'source_sha256':sha(Path(__file__)),'scope':'synthetic proper loss/gradient, no N9data or fits yet'});print('N9 gradient fixtures passed',max(errors),flush=True)
if __name__=='__main__':main()
