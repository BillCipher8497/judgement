"""Additional manual true-posterior one-step payoff check for the fixed example."""
import os,sys,json,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parent;O=H/'results';sys.path.insert(0,str(R/'N9'))
import n9lib as B
from reference import direct_posterior,legal,winner
from bounded_ta import ta_card
from judgement.cfr.heuristics import myopic_bid
from common import save,sha,stamp,log
def main():
    assert time.time()<float(os.environ['JUDGEMENT_JOB_DEADLINE']);a=json.loads((O/'certificate.json').read_text());g=B.game(a['tag']);hero=a['hero'];hist=tuple(a['public_history']);_,worlds=direct_posterior(g,hero,tuple(a['own_initial_hand']),hist,myopic_bid,ta_card);values=[]
    for w in worlds:
        row=[]
        for forced in (4,6):
            left=[list(h) for h in w];trick=[];leader=0;won=[0]*4
            for c in hist[4:]:
                p=(leader+len(trick))%4;assert c in legal(g,left[p],trick);left[p].remove(c);trick.append((p,c))
                if len(trick)==4:leader=winner(g,trick);won[leader]+=1;trick=[]
            h=hist;first=True
            while len(h)<16:
                p=(leader+len(trick))%4;cc=legal(g,left[p],trick)
                if first:assert p==hero;c=forced;first=False
                elif p==hero:assert len(cc)==1;c=cc[0]
                else:c=cc[0] if len(cc)==1 else ta_card(g,p,w[p],h)
                assert c in cc;left[p].remove(c);trick.append((p,c));h+=(c,)
                if len(trick)==4:leader=winner(g,trick);won[leader]+=1;trick=[]
            row.append(10+hist[hero] if won[hero]==hist[hero] else 0)
        values.append(row)
    assert values==[[10,10]]*3;assert a['secondary_existing_private_actions']['exact_posterior_score_sums']==[30,30];path=R/'N12/results'/a['tag']/f"{a['key_id']:07d}.npz";assert sha(path)==a['secondary_existing_private_actions']['source_sha256'];save(O/'secondary_action_check.json',{'utc':stamp(),'n_exact_posterior_worlds':3,'actions':[4,6],'manual_world_action_scores':values,'exact_private_values':['10','10'],'regret_of_either_current_action_exact':'0','source_sha256':sha(Path(__file__)),'certificate_sha256':sha(O/'certificate.json'),'scope':'Exact same fixed illustrative toy context, no samplingCI/population/game-strength claim; pending external'});log(H,'N15manual3worldx2action check passed: both exactprivatevalues10,one-step regret0; belief-familyinfimumpositive remains separate, pendingexternal/local.')
if __name__=='__main__':main()
