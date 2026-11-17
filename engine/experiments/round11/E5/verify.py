"""Separate integer-vector/checkpoint/map audit of the fresh forward replay."""
import gzip
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
HERE = Path(__file__).resolve().parent; ENGINE = HERE.parents[2]; OUT = HERE/'results'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''): h.update(block)
    return h.hexdigest()


def main():
    checks = {}
    for trump, expected in [(0, Fraction(574247, 92400)), (-1, Fraction(109709, 15400))]:
        result = json.loads((OUT/f'replay_{trump}.json').read_text())
        scores = (OUT/result['raw_hero_scores_file']).read_bytes()
        assert len(scores) == 369600 and set(scores) <= {0, 10, 11, 12, 13}
        assert hashlib.sha256(scores).hexdigest() == result['raw_hero_scores_sha256']
        fraction = Fraction(sum(scores), len(scores))
        assert fraction == expected == Fraction(result['exact_feasible_upper'])
        assert sum(scores) == result['hero_score_sum']
        for rel, expected_hash in result['source_sha256'].items():
            source = (ENGINE/rel.replace('\\', '/')).read_bytes().replace(b'\r\n', b'\n')
            assert hashlib.sha256(source).hexdigest() == expected_hash
        checkpoints = [json.loads(x) for x in (OUT/f'replay_{trump}_checkpoints.jsonl').read_text().splitlines()]
        previous = total = 0
        for row in checkpoints:
            count = row['completed']; assert count > previous and count-previous in [600, 1000]
            total += sum(scores[previous:count]); assert total == row['hero_score_sum']
            previous = count
        assert previous == 369600 and total == sum(scores)
        metadata = json.loads((OUT/f'private_policy_{trump}.json').read_text())
        path = OUT/f'private_policy_{trump}.bin.gz'; assert digest(path) == metadata['mapping_sha256']
        rows = 0
        with gzip.open(path, 'rb') as f:
            assert f.readline() == b'JUDGEMENT_PRIVATE_POLICY_V1\n'
            while header := f.read(2):
                assert len(header) == 2
                n, action = header; key = f.read(n)
                assert len(key) == n and key[0] in [1, 2, 3]
                hand = list(key[1:4]); assert len(set(hand)) == 3 and sorted(hand) == hand and all(c < 12 for c in hand)
                assert action in hand
                hist = key[4:]; assert len(hist) >= 4 and all(b <= 3 for b in hist[:4]) and sum(hist[:4]) != 3
                rows += 1
                if rows % 1000000 == 0:
                    (OUT/'verification_progress.json').write_text(json.dumps({'trump': trump, 'private_rows_checked': rows}))
        assert rows == metadata['n_private_keys'] == result['mapping']['n_private_keys']
        checks[str(trump)] = {'n_exact_deals': len(scores), 'hero_score_sum': sum(scores),
                              'exact_feasible_upper': str(fraction), 'map_rows_checked': rows,
                              'raw_score_sha256': result['raw_hero_scores_sha256'], 'pass': True}
    packet = {'pass': True, 'checks': checks, 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'method': 'Fresh byte-vector integer sums, every block checkpoint, source hashes and all portable private-map rows; no optimizer/tree values',
              'limits': 'Same investigator; original fixed hero/bidder definitions retained, private keys exported from source-fingerprinted cache; no external replay or solved hidden optimum; exact complete populations, no sampling CI'}
    (OUT/'verification.json').write_text(json.dumps(packet, indent=2)+'\n')
    print(json.dumps(packet, indent=2))


if __name__ == '__main__':
    deadline = datetime(2026, 11, 17, 4, 0, tzinfo=timezone.utc).timestamp(); assert time.time() < deadline
    cut = threading.Timer(deadline-time.time(), lambda: os._exit(124)); cut.daemon = True; cut.start()
    try: main()
    finally: cut.cancel()
