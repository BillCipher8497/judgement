"""Post-run explicit world/contingent-table certificates, including tie sensitivity."""
from run import *
from fractions import Fraction
def main():
    verified=json.loads((O/'verification.json').read_text());complete=json.loads((O/'primary_complete.json').read_text());assert len(verified['checks'])==2
    guard(complete['sources'])
    inventory=[]
    for s in complete['summaries']:
        tag=s['tag'];trump=0 if tag.endswith('t0') else -1;g=N6.ToyGame(N=4,S=3,R=4,n=3,trump=trump);d,_=corpus(tag);folder=O/tag;selected=json.loads((folder/'selection.json').read_text())['indices']
        for i in selected:
            guard(complete['sources'])
            r=json.loads((folder/f'{i:07d}.json').read_text())
            if not any(a>b for a,b in zip(r['PI_Q_sums'],r['BR_Q_sums'])):continue
            hero,hand,hist=N6.decode(d['keys'][i]);_,worlds=V.direct_posterior(g,hero,hand,hist,V.myopic_bid,N6.ta_card);worlds=sorted(worlds,key=lambda w:V.deal_index(g,w));worldids=[V.deal_index(g,w) for w in worlds]
            expected=d['support_deal_index'][int(d['support_offset'][i]):int(d['support_offset'][i+1])].astype(int).tolist();assert worldids==expected
            tables=[]
            for action,pi,br in zip(r['actions'],r['PI_Q_sums'],r['BR_Q_sums']):
                groups={};constant=0;terminal_worlds=[]
                for wid,hands in zip(worldids,worlds):
                    prefix,choices,value=V.advance(g,hands,hero,hist,action)
                    if value is not None:constant+=value;terminal_worlds.append({'deal_index':wid,'score':value});continue
                    scores=[]
                    for card in choices:
                        _,tail,score=V.advance(g,hands,hero,prefix,card);assert not tail and score is not None;scores.append(score)
                    groups.setdefault((prefix,choices),[]).append({'deal_index':wid,'scores_by_card':scores})
                explicit=[];totalpi=constant;totalbr=constant
                for (prefix,choices),matrix in groups.items():
                    columns=[sum(row['scores_by_card'][j] for row in matrix) for j in range(len(choices))];upper=sum(max(row['scores_by_card']) for row in matrix);private=max(columns);totalpi+=upper;totalbr+=private
                    explicit.append({'future_public_history':prefix,'future_legal_cards':choices,'world_score_rows':matrix,'private_column_sums':columns,'private_optimal_card':choices[max(range(len(columns)),key=lambda j:columns[j])],'PI_worldwise_max_sum':upper,'private_common_action_max_sum':private,'illegal_fusion_optimism_sum':upper-private})
                assert totalpi==pi and totalbr==br
                tables.append({'current_card':action,'terminal_constant_sum':constant,'terminal_worlds':terminal_worlds,'future_private_infoset_tables':explicit,'PI_Q_sum':totalpi,'BR_Q_sum':totalbr})
            pmax=max(r['PI_Q_sums']);bmax=max(r['BR_Q_sums']);top=[j for j,a in enumerate(r['PI_Q_sums']) if a==pmax];n=r['support_n']
            bestloss=bmax-max(r['BR_Q_sums'][j] for j in top);worstloss=bmax-min(r['BR_Q_sums'][j] for j in top)
            payload={'condition':tag,'key_index':i,'hero':hero,'own_hand':hand,'public_history':hist,'support_n':n,'worlds':[{'deal_index':wid,'hands':hands} for wid,hands in zip(worldids,worlds)],'actions':r['actions'],'PI_Q_sums':r['PI_Q_sums'],'BR_Q_sums':r['BR_Q_sums'],'PI_optimal_cards':[r['actions'][j] for j in top],'BR_optimal_cards':[r['actions'][j] for j,b in enumerate(r['BR_Q_sums']) if b==bmax],'fixed_lower_card_PIMC_action':r['PIMC_action'],'fixed_tie_loss_exact':str(Fraction(r['first_action_loss_sum'],n)),'best_PI_tie_loss_exact':str(Fraction(bestloss,n)),'worst_PI_tie_loss_exact':str(Fraction(worstloss,n)),'tie_robust_strict_loss':bestloss>0,'tables':tables,'sampling_CI':None,'verdict':'Exact finite computational certificate, independently replayed support/table; pending Claude external audit','source_sha256':sha(Path(__file__))}
            target=O/'certificates'/f'{tag}_{i:07d}.json';target.parent.mkdir(exist_ok=True);save(target,payload)
            inventory.append({k:payload[k] for k in ('condition','key_index','hero','own_hand','public_history','support_n','actions','PI_Q_sums','BR_Q_sums','PI_optimal_cards','BR_optimal_cards','fixed_tie_loss_exact','best_PI_tie_loss_exact','worst_PI_tie_loss_exact','tie_robust_strict_loss')}|{'file':str(target.relative_to(O)),'sha256':sha(target)})
            N6.clear_mid()
    save(O/'certificates.json',{'utc':stamp(),'all_positive_any_action_gap_keys':inventory,'n_certificates':len(inventory),'verification_sha256':sha(O/'verification.json'),'source_sha256':sha(Path(__file__)),'scope':'Post-run explicit certificates/tie diagnostics, not new predeclared contrasts or prevalences. All original selectors/values/decoder fixed.'})
    print(json.dumps(inventory),flush=True)
if __name__=='__main__':main()
