"""Fresh forward policy replay, without cached transitions/values or optimization.

Only the saved private key -> action association is exported from the derived
cache. All deals, legal moves, trick winners and hero scores are reconstructed
here. The fixed hero/bidder definitions are inherited; they are the game being
evaluated, not an optimizing evaluator. No solver/CFR/value routine is called.
"""
import gc
import gzip
import hashlib
import json
import os
import pickle
import struct
import sys
import threading
import time
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[2]
D3 = HERE.parents[1]/'round9/D3'
sys.path[:0] = [str(ENGINE), str(D3)]
from judgement.cfr.toy import ToyGame
from judgement.cfr.heuristics import myopic_bid
from bounded_ta import ta_card, clear_mid

OUT = Path(os.environ.get('JUDGEMENT_REPLAY_OUT', str(HERE/'results'))).resolve()
EXPECTED = {0: Fraction(574247, 92400), -1: Fraction(109709, 15400)}
DEADLINE = float(os.environ.get('JUDGEMENT_REPLAY_DEADLINE',
    datetime(2026, 11, 17, 4, 0, tzinfo=timezone.utc).timestamp()))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''): h.update(block)
    return h.hexdigest()


def save(p, item):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(item, indent=2)+'\n')
    os.replace(temp, p)


def log(message):
    with (HERE/'STATUS.txt').open('a', encoding='utf-8') as f:
        f.write(datetime.now(timezone.utc).isoformat()+' '+message+'\n')
    print(message, flush=True)


def independent_deals():
    deck = tuple(range(12))
    for a in combinations(deck, 3):
        r1 = tuple(c for c in deck if c not in a)
        for b in combinations(r1, 3):
            r2 = tuple(c for c in r1 if c not in b)
            for c in combinations(r2, 3):
                yield (a, b, c, tuple(x for x in r2 if x not in c))


def legal_cards(hand, trick):
    if not trick: return hand[:]
    following = [c for c in hand if c//4 == trick[0][1]//4]
    return following or hand[:]


def trick_winner(trick, trump):
    led = trick[0][1]//4
    def key(play):
        c = play[1]; suit = c//4
        category = 2 if trump != -1 and suit == trump else int(suit == led)
        return category, c % 4
    return max(trick, key=key)[0]


def load_policy(trump):
    policy_file = D3/'results'/f'review_pure_trump_{trump}.npz'
    old_audit = json.loads((D3.parent/'reviews/results/D3_exact_upper_audit.json').read_text())[str(trump)]
    assert old_audit['policy_sha256'] == sha(policy_file), 'Saved policy changed since original upper packet'
    store = OUT
    published = HERE/'results'
    if OUT != published and (published/f'private_policy_{trump}.bin.gz').exists() and (published/f'private_policy_{trump}.json').exists():
        store = published
    metadata_file = store/f'private_policy_{trump}.json'
    portable = store/f'private_policy_{trump}.bin.gz'
    source_paths = [D3/'compact_tree.py', D3/'bounded_ta.py',
                    ENGINE/'judgement/cfr/toy.py', ENGINE/'judgement/cfr/heuristics.py']
    fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in source_paths)).hexdigest()
    canonical_fingerprint = hashlib.sha256(b''.join(p.read_bytes().replace(b'\r\n', b'\n') for p in source_paths)).hexdigest()
    if portable.exists() and metadata_file.exists():
        meta = json.loads(metadata_file.read_text())
        assert meta['policy_sha256'] == sha(policy_file)
        assert meta['mapping_sha256'] == sha(portable)
        assert meta['canonical_cache_source_sha256'] == canonical_fingerprint
    else:
        meta = json.loads((D3/'results'/f'tree_cache_{trump}.json').read_text())
        assert meta['source_sha256'] == fingerprint
        cache = Path(meta['derived_cache']); assert cache.exists(), 'Need original private-key cache or portable mapping'
        with cache.open('rb') as f: tree = pickle.load(f)
        assert (tree.g.N, tree.g.S, tree.g.R, tree.g.n, tree.g.trump, tree.g.hit_base, tree.g.hook, tree.g.util) == (4, 3, 4, 3, trump, 10, True, 'raw')
        actions = np.load(policy_file)['actions']
        assert len(actions) == tree.n_is
        assert np.all(tree.is_valid[np.arange(tree.n_is), actions])
        temp = portable.with_suffix('.tmp')
        with gzip.open(temp, 'wb', compresslevel=1) as out:
            out.write(b'JUDGEMENT_PRIVATE_POLICY_V1\n')
            for i, action in enumerate(actions):
                p, hand, hist = tree.is_key[i]
                assert p != 0 and len(hand) == 3
                key = bytes((p, *hand, *hist))
                card = hand[int(action)]
                out.write(struct.pack('BB', len(key), card)); out.write(key)
        os.replace(temp, portable)
        meta = {'trump': trump, 'n_private_keys': int(tree.n_is),
                'policy_sha256': sha(policy_file), 'mapping_sha256': sha(portable),
                'derived_cache_sha256': sha(cache),
                'canonical_cache_source_sha256': canonical_fingerprint,
                'cache_source_sha256': fingerprint,
                'semantics': 'private seat+sorted own initial hand+full public bids/cards -> card',
                'cache_use': 'keys/action validity ONLY; no edges, terminal values, roots or optimization read'}
        save(metadata_file, meta)
        del tree, actions; gc.collect()
    policy = {}
    with gzip.open(portable, 'rb') as f:
        assert f.readline() == b'JUDGEMENT_PRIVATE_POLICY_V1\n'
        while header := f.read(2):
            assert len(header) == 2
            length, card = header; key = f.read(length)
            assert len(key) == length and key not in policy
            assert card in key[1:4]
            policy[key] = card
    assert len(policy) == meta['n_private_keys']
    assert meta['trump'] == trump
    return policy, meta


def replay(hands, g, policy):
    bids = []
    for p in range(4): bids.append(myopic_bid(g, p, hands[p], tuple(bids)))
    assert len(bids) == 4 and sum(bids) != 3 and all(0 <= b <= 3 for b in bids)
    left = [list(h) for h in hands]
    history = bytes(bids); tricks = [0]*4; leader = 0; trick = []
    lookups = 0
    for _ in range(12):
        remaining = len(left[0])+int(any(p == 0 for p, _ in trick))
        need = bids[0]-tricks[0]
        if need < 0 or need > remaining: return 0, lookups
        p = (leader+len(trick)) % 4
        legal = legal_cards(left[p], trick)
        if len(legal) == 1: card = legal[0]
        elif p == 0: card = ta_card(g, 0, hands[0], tuple(history))
        else:
            # Only the acting spoiler's private hand and public sequence occur
            # in this lookup, never opponents' or hero's unrevealed cards.
            key = bytes((p, *hands[p]))+history
            card = policy[key]; lookups += 1
        assert card in legal
        left[p].remove(card); history += bytes((card,)); trick.append((p, card))
        if len(trick) == 4:
            leader = trick_winner(trick, g.trump); tricks[leader] += 1; trick = []
    assert not any(left) and not trick and sum(tricks) == 3
    return (10+bids[0] if tricks[0] == bids[0] else 0), lookups


def run(trump):
    policy, metadata = load_policy(trump)
    g = ToyGame(N=4, S=3, R=4, n=3, trump=trump)
    paths = [Path(__file__), D3/'bounded_ta.py', ENGINE/'judgement/cfr/heuristics.py',
             ENGINE/'judgement/cfr/toy.py']
    raw_fingerprints = {str(p.relative_to(ENGINE)): sha(p) for p in paths}
    fingerprints = {str(p.relative_to(ENGINE)): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in paths}
    checkpoints = OUT/f'replay_{trump}_checkpoints.jsonl'
    rows = []
    if checkpoints.exists():
        lines = checkpoints.read_bytes().splitlines(keepends=True)
        for index, line in enumerate(lines):
            try: rows.append(json.loads(line))
            except ValueError:
                assert index == len(lines)-1
                tail = checkpoints.with_suffix('.incomplete_tail')
                assert not tail.exists(), 'Inspect previous incomplete checkpoint tail'
                tail.write_bytes(line); checkpoints.write_bytes(b''.join(lines[:index])); break
    completed = rows[-1]['completed'] if rows else 0
    total = rows[-1]['hero_score_sum'] if rows else 0
    lookups = rows[-1]['private_policy_lookups'] if rows else 0
    if rows:
        assert rows[-1]['source_sha256'] == fingerprints
        assert rows[-1]['policy_mapping_sha256'] == metadata['mapping_sha256']
    raw_scores = OUT/f'hero_scores_{trump}.bin'
    if raw_scores.exists():
        raw = raw_scores.read_bytes()
        assert len(raw) >= completed
        if len(raw) > completed:
            tail = OUT/f'hero_scores_{trump}_uncheckpointed_{time.time_ns()}.bin'
            tail.write_bytes(raw[completed:]); raw_scores.write_bytes(raw[:completed])
        assert sum(raw[:completed]) == total
    else:
        assert completed == 0, 'Completed score-vector checkpoint is missing'
        raw_scores.write_bytes(b'')
    batch = bytearray()
    tick = time.perf_counter()
    for index, hands in enumerate(independent_deals()):
        if index < completed: continue
        if time.time() >= DEADLINE: raise TimeoutError('E5 hard stop')
        score, used = replay(hands, g, policy)
        total += score; lookups += used; completed = index+1
        batch.append(score)
        if completed % 1000 == 0 or completed == 369600:
            with raw_scores.open('ab') as f: f.write(batch); f.flush()
            batch.clear()
            packet = {'completed': completed, 'hero_score_sum': total,
                      'private_policy_lookups': lookups, 'source_sha256': fingerprints,
                      'source_raw_sha256': raw_fingerprints,
                      'source_hash_convention': 'Python source CRLF normalized to LF; raw SHA also retained',
                      'policy_mapping_sha256': metadata['mapping_sha256'],
                      'checkpoint_utc': datetime.now(timezone.utc).isoformat()}
            with checkpoints.open('a', encoding='utf-8') as f: f.write(json.dumps(packet)+'\n')
            clear_mid()
            if completed % 10000 == 0: log(f'trump{trump}: fresh replay{completed}/369600deals')
    assert completed == 369600
    assert len(raw_scores.read_bytes()) == completed and sum(raw_scores.read_bytes()) == total
    fraction = Fraction(total, completed)
    result = {'trump': trump, 'n_exact_deals': completed, 'hero_score_sum': total,
              'exact_feasible_upper': str(fraction), 'expected': str(EXPECTED[trump]),
              'match': fraction == EXPECTED[trump], 'private_policy_lookups': lookups,
              'raw_hero_scores_file': raw_scores.name, 'raw_hero_scores_sha256': sha(raw_scores),
              'seconds_current_segment': time.perf_counter()-tick,
              'mapping': metadata, 'source_sha256': fingerprints,
              'source_raw_sha256': raw_fingerprints,
              'source_hash_convention': 'Python source CRLF normalized to LF; raw SHA also retained',
              'method': 'Independent forward deals/legality/trick/scoring; original fixed hero/bidder definitions; no optimizer/cached edges/values',
              'uncertainty': 'Complete finite chance population, exact rational; no sampling CI; feasible upper, not solved optimum'}
    save(OUT/f'replay_{trump}.json', result)
    assert result['match'], 'Fresh replay failed: report discrepancy, do not alter candidate'
    log(json.dumps(result))
    del policy; gc.collect(); clear_mid()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert time.time() < DEADLINE
    timer = threading.Timer(DEADLINE-time.time(), lambda: os._exit(124)); timer.daemon = True; timer.start()
    try:
        for trump in [0, -1]: run(trump)
    finally: timer.cancel()


if __name__ == '__main__': main()
