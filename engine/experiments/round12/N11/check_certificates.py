"""Third, algebra-only JSON certificate check; imports no game or primary solver."""
import json,hashlib
from pathlib import Path
from fractions import Fraction
from datetime import datetime,timezone
H=Path(__file__).resolve().parent;O=H/'results'
def read(p):return json.loads(p.read_text())
def main():
    inv=read(O/'certificates.json');assert inv['n_certificates']==7;checked=[]
    for c in inv['all_positive_any_action_gap_keys']:
        path=O/c['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==c['sha256'];w=read(path);n=w['support_n'];ids={r['deal_index'] for r in w['worlds']};assert len(ids)==n
        payoff=10+w['public_history'][w['hero']]
        for world in w['worlds']:
            assert world['hands'][w['hero']]==w['own_hand']
            assert sorted(card for hand in world['hands'] for card in hand)==list(range(12)) and all(len(hand)==3 for hand in world['hands'])
        pi=[];br=[]
        for table in w['tables']:
            seen=[];constant=sum(r['score'] for r in table['terminal_worlds']);seen.extend(r['deal_index'] for r in table['terminal_worlds']);assert constant==table['terminal_constant_sum'];a=constant;b=constant
            for group in table['future_private_infoset_tables']:
                rows=group['world_score_rows'];seen.extend(r['deal_index'] for r in rows);columns=[sum(r['scores_by_card'][j] for r in rows) for j in range(len(group['future_legal_cards']))]
                assert columns==group['private_column_sums']
                assert all(v in (0,payoff) for r in rows for v in r['scores_by_card'])
                upper=sum(max(r['scores_by_card']) for r in rows);private=max(columns)
                assert upper==group['PI_worldwise_max_sum'] and private==group['private_common_action_max_sum'] and upper-private==group['illegal_fusion_optimism_sum']>=0
                a+=upper;b+=private
            assert len(seen)==n and set(seen)==ids and a==table['PI_Q_sum'] and b==table['BR_Q_sum'];pi.append(a);br.append(b)
        assert pi==w['PI_Q_sums'] and br==w['BR_Q_sums'];top=[j for j,p in enumerate(pi) if p==max(pi)];best=max(br);loss=best-br[top[0]]
        assert str(Fraction(loss,n))==w['fixed_tie_loss_exact']
        assert str(Fraction(best-max(br[j] for j in top),n))==w['best_PI_tie_loss_exact']
        assert str(Fraction(best-min(br[j] for j in top),n))==w['worst_PI_tie_loss_exact'];checked.append(c['file'])
    witness=read(O/'certificates/S3R4n3_t0_0507870.json')
    assert witness['support_n']==14 and witness['hero']==1 and witness['own_hand']==[4,6,10] and witness['public_history']==[0,0,2,0,1]
    assert witness['actions']==[4,6,10] and witness['PI_Q_sums']==[140,140,140] and witness['BR_Q_sums']==[100,140,140]
    assert witness['fixed_tie_loss_exact']=='20/7' and witness['best_PI_tie_loss_exact']=='0' and not witness['tie_robust_strict_loss']
    result={'utc':datetime.now(timezone.utc).isoformat(),'n_certificates':len(checked),'all_world_partitions_leaf_scores_PI_max_private_column_sums_and_tie_ranges_match':True,'fourteen_world_tie_sensitive_witness_pass':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'verdict':'Algebra-only separate Codex check; pending external audit; policy reach/transitions separately verified by direct enumerator.'}
    (O/'certificate_algebra_check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
