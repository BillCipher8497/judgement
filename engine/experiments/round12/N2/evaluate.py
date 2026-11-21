"""N2 exact finite-population joint KL evaluation prototype; chunk checkpoints."""
import json,math,time,statistics
from pathlib import Path
import numpy as np
from corpus import structural_and_bid_KL
from neural import probabilities
from joint import product_kl
from fit import params,save,check_deadline
from export import atomic_npz,digest

def interval(values):
    assert len(values)==5;m=statistics.mean(values);h=2.7764451051977987*statistics.stdev(values)/math.sqrt(5)
    return {'mean':m,'ci95':[m-h,m+h],'n_optimization_seeds':5,'method':'t95%CI df4 over fitting seeds only; complete chance population per fixed fit'}

def evaluate(corpus,model_folder,output,freeze,deadline,source_check,status,myopic_bid):
    output=Path(output);output.mkdir(exist_ok=True);models=json.loads((Path(model_folder)/'frozen_models.json').read_text());assert models['source_freeze_sha256']==freeze
    model_hash=digest(Path(model_folder)/'frozen_models.json');posterior_hash=digest(corpus.folder/'posterior_supports.npz')
    assert posterior_hash==models['posterior_sha256']
    parameters=[];alphas=[]
    for s,r in enumerate(models['models']):
        path=Path(model_folder)/f's{s}'/'selected.npz';assert digest(path)==r['selected_model_sha256'];parameters.append(params(path));alphas.append(r['alpha'])
    n=len(corpus.keys)
    for start in range(0,n,1024):
        check_deadline(deadline);source_check();p=output/f'{start:07d}.npz';receipt=p.with_suffix('.json');end=min(start+1024,n)
        if receipt.exists():
            r=json.loads(receipt.read_text());assert r['freeze_sha256']==freeze and r['model_packet_sha256']==model_hash and r['posterior_sha256']==posterior_hash and r['arrays_sha256']==digest(p) and r['start']==start and r['end']==end;continue
        if p.exists():p.rename(p.with_name(p.name+'.orphan_'+str(time.time_ns())))
        ids=np.arange(start,end);pack=corpus.pack(ids);pred=[(1-.999*a)*pack['P']+.999*a*probabilities(pack['X'],pack['M'],pack['P'],v) for a,v in zip(alphas,parameters)]
        base=np.zeros((len(ids),3));worlds=np.zeros((len(ids),3),np.uint32);bid_integer=np.zeros((len(ids),2),np.uint64);net=np.zeros((len(ids),5));raw_net=np.zeros_like(net);card_loss=np.zeros((len(ids),6))
        for k,i in enumerate(ids):
            check_deadline(deadline);x,mask,prop,target,public=corpus.row(int(i));reach=int(corpus.count[i]);baseline,allowed=structural_and_bid_KL(corpus.g,public['observation'],public,reach,myopic_bid)
            base[k]=[baseline['uniform_KL_float'],baseline['BI_KL_float'],baseline['hard_bid_KL_float']]
            worlds[k]=[baseline['feasible_world_count'],baseline['hard_bid_consistent_world_count'],reach]
            bid_integer[k]=[baseline['BI_integer_partition'],baseline['BI_max_integer_weight']]
            assert base[k,2]<=base[k,1]+1e-10<=base[k,0]+2e-10
            a,b=pack['offset'][k:k+2];counts=corpus.data['holder_counts'][i,list(public['cards']),:]
            if b>a:card_loss[k,0]=float(-np.sum(pack['T'][a:b]*np.log(np.maximum(prop,1e-300)))/(b-a))
            for s in range(5):
                result=product_kl(reach,counts,public['capacities'],allowed,pred[s][a:b,:3]);net[k,s]=result['KL_float'];raw_net[k,s]=result['raw_KL_float']
                if b>a:card_loss[k,s+1]=float(-np.sum(pack['T'][a:b]*np.log(np.maximum(pred[s][a:b],1e-300)))/(b-a))
        atomic_npz(p,group_index=ids,baseline_KL=base,world_counts=worlds,BI_integer_partition_and_maxweight=bid_integer,net_KL=net,raw_net_KL=raw_net,raw_marginal_card_logloss=card_loss,reach=corpus.count[ids],split=corpus.split[ids])
        save(receipt,{'start':start,'end':end,'freeze_sha256':freeze,'model_packet_sha256':model_hash,'posterior_sha256':posterior_hash,'arrays_sha256':digest(p),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        status(f'JointKLsaved{end}/{n}privateinfosets')
    total={label:{'denominator':0,'baseline':np.zeros(3),'net':np.zeros(5),'raw_net':np.zeros(5),'marginal':np.zeros(6),'n_keys':0,'hands':set()} for label in ('all','train','validation','test')}
    seen=0;hashes={}
    from export import decode
    for start in range(0,n,1024):
        p=output/f'{start:07d}.npz';r=json.loads(p.with_suffix('.json').read_text());assert digest(p)==r['arrays_sha256'];hashes[p.name]=r['arrays_sha256'];a=np.load(p);assert a['group_index'][0]==seen;seen+=len(a['group_index'])
        for label,split in (('all',None),('train',0),('validation',1),('test',2)):
            take=np.ones(len(a['reach']),bool) if split is None else a['split']==split;w=a['reach'][take].astype(float);t=total[label]
            t['denominator']+=int(a['reach'][take].sum());t['n_keys']+=int(take.sum());t['baseline']+=np.sum(a['baseline_KL'][take]*w[:,None],axis=0);t['net']+=np.sum(a['net_KL'][take]*w[:,None],axis=0);t['raw_net']+=np.sum(a['raw_net_KL'][take]*w[:,None],axis=0)
            t['marginal']+=np.sum(a['raw_marginal_card_logloss'][take]*w[:,None],axis=0)
            t['hands'].update(decode(corpus.keys[int(i)])[1] for i in a['group_index'][take])
    assert seen==n
    summaries={}
    for label,t in total.items():
        den=t['denominator'];base=t['baseline']/den;values=(t['net']/den).tolist();raw=(t['raw_net']/den).tolist()
        marginal=t['marginal']/den;shares=None if base[0]==0 else {'hard_bid':float(1-base[2]/base[0]),'soft_BI':float(1-base[1]/base[0]),'net_by_seed':[float(1-v/base[0]) for v in values],'net_seed_summary':interval([float(1-v/base[0]) for v in values]),'scope':'Descriptive share of mean uniform joint KL removed; not additive, negative shares retained, no game-score meaning.'}
        summaries[label]={'n_private_keys':t['n_keys'],'n_reached_own_hands':len(t['hands']),'exact_reach_weight':den,'uniform_KL':float(base[0]),'prop_joint_KL':float(base[0]),'BI_KL':float(base[1]),'hard_bid_KL':float(base[2]),'share_uniform_KL_removed':shares,
            'raw_marginal_card_logloss':{'capacity_prop':float(marginal[0]),'net_by_seed':marginal[1:].tolist(),'net_seed_summary':interval(marginal[1:].tolist()),'scope':'Raw capacity/net per-card probabilities; not marginals of their structurally conditioned joint laws. Zero-card keys have zero loss but retain denominator.'},
            'net_KL_by_seed_exact_population_float':values,'raw_unclipped_net_KL_by_seed':raw,'net_seed_summary':interval(values),'primary_net_minus_BI':interval([v-base[1] for v in values]),
            'scope':'Full fixed-condition reached population for this split; no chance-sampling CI. Optimization interval only. All/train/validation descriptive; test primary.'}
    result={'condition':f'S{corpus.g.S}R{corpus.g.R}n{corpus.g.n}_t{corpus.g.trump}','summaries':summaries,'raw_chunks_sha256':hashes,'source_freeze_sha256':freeze,
        'limitations':'Fixed private all-TA/myopic profile, tiny all-dealt game. Summary features and factorized-joint family may lose information. Own-hand/key split shares latent deals across actors. New algebra pending external audit; no team score/bound or strategy-fusion claim.',
        'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    save(output.parent/'joint_summary.json',result);return result
