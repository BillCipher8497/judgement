"""Held evaluation after all120trajectories/240selected fits freeze."""
from n9lib import *
from scipy.stats import t
def ci(v):
    v=np.array(v,float);m=float(v.mean());h=float(t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)))
    return {'mean':m,'ci95':[m-h,m+h],'n_optimizer_seeds':len(v),'method':'paired where contrasted; t95%df4, optimization seeds only on known deterministic held-key subset'}
def main():
    complete=json.loads((O/'training_complete.json').read_text());assert complete['n_trajectories']==120;results=[]
    for tag in TAGS:
        rs=rows(load(O/tag/'data.npz'));test=[r for r in rs if r.split==2];w=np.array([r.count for r in test]);baseline=float(sum(r.count*r.BI for r in test)/w.sum())
        for cfg in CONFIGS:
            for loss in LOSSES:
                for seed in range(5):
                    folder=O/tag/'models'/f'{cfg}_{loss}_s{seed}';receipt=folder/'frozen.json';assert sha(receipt)==complete['receipts'][str(receipt.relative_to(O))];f=json.loads(receipt.read_text())
                    for budget in (20,80):
                        check();path=folder/f'selected{budget}.npz';assert sha(path)==f['models'][str(budget)]['sha256'];z=load(path);p=[z[f'p{i}'] for i in range(len(z))];mean,values=evaluate(test,p,cfg=='rich_mlp');scores=np.concatenate([score(r,p,cfg=='rich_mlp') for r in test]);npz(folder/f'held_{budget}.npz',KL=values,logits=scores,key_ids=np.array([r.kid for r in test]))
                        hands={};
                        for r,value in zip(test,values):hands.setdefault(str(decode(r.key)[1]),[]).append((r.count,float(value)))
                        perhand={h:sum(n*v for n,v in pairs)/sum(n for n,v in pairs) for h,pairs in hands.items()}
                        results.append({'tag':tag,'config':cfg,'loss':loss,'seed':seed,'budget':budget,'selected_epoch':f['models'][str(budget)]['selected_epoch'],'joint_KL':mean,'BI_joint_KL':baseline,'minus_BI':mean-baseline,'n_test_keys':len(test),'n_test_hands':len(hands),'test_reach':int(w.sum()),'per_hand_KL':perhand,'model_sha256':sha(path)})
    def values(tag,cfg,loss,budget):return [next(r['joint_KL'] for r in results if (r['tag'],r['config'],r['loss'],r['budget'],r['seed'])==(tag,cfg,loss,budget,s)) for s in range(5)]
    summaries={};objective={};duration={};features={}
    for tag in TAGS:
        for cfg in CONFIGS:
            for loss in LOSSES:
                for budget in (20,80):summaries[f'{tag}/{cfg}/{loss}/{budget}']=ci(values(tag,cfg,loss,budget))
                duration[f'{tag}/{cfg}/{loss}']=ci(np.array(values(tag,cfg,loss,80))-values(tag,cfg,loss,20))
            for budget in (20,80):objective[f'{tag}/{cfg}/{budget}']=ci(np.array(values(tag,cfg,'joint',budget))-values(tag,cfg,'marginal',budget))
        for loss in LOSSES:
            for budget in (20,80):features[f'{tag}/{loss}/{budget}']=ci(np.array(values(tag,'rich_mlp',loss,budget))-values(tag,'summary_mlp',loss,budget))
    save(O/'evaluation.json',{'utc':stamp(),'rows':results,'arm_summaries':summaries,'joint_minus_marginal':objective,'long_minus_short':duration,'rich_minus_summary':features,'primary':objective['S3R4n3_t-1/summary_mlp/80'],'scope':'known deterministic hand-balanced key subsets, complete exact support perkey, no chance CI;5paired optimizer seeds only; no fullgame/factor-family-optimum/cause/novelty inference','source_sha256':sha(Path(__file__)),'training_sha256':sha(O/'training_complete.json')})
    log(H,'Held N9evaluation complete: '+json.dumps(objective['S3R4n3_t-1/summary_mlp/80'])+'; allcells/nulls retained, pending separate audit/external validation.')
if __name__=='__main__':main()
