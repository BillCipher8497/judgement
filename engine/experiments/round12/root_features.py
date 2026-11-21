"""Strict private-observation encoder for the frozen Round12 root benchmark."""
import numpy as np

SCHEMA = {'seat', 'hand', 'history', 'n', 'S', 'R', 'trump', 'legal'}
WIDTH = 241


def encode(obs):
    assert set(obs) == SCHEMA, 'Only private observation schema accepted'
    p, h, history, n, S, R, trump, legal = (obs[k] for k in ('seat', 'hand', 'history', 'n', 'S', 'R', 'trump', 'legal'))
    assert p in (1, 2, 3) and R == 4 and (S, n) in ((2, 2), (3, 3)) and trump in (0, -1)
    assert h == sorted(set(h)) and len(h) == n and all(0 <= c < S*R for c in h)
    assert 4 <= len(history) <= 4+4*n and len(legal) >= 2 and legal == sorted(set(legal))
    assert all(0 <= a < n for a in legal)
    row = np.zeros(WIDTH); cursor = 0
    row[h] = 1.; cursor += 12
    for slot in range(3):
        row[cursor+(h[slot] if slot < n else 12)] = 1.; cursor += 13
    for b in history[:4]:
        assert 0 <= b <= n
        row[cursor+b] = 1.; cursor += 5
    for slot in range(12):
        c = history[4+slot] if 4+slot < len(history) else 12
        assert c == 12 or 0 <= c < S*R
        row[cursor+c] = 1.; cursor += 13
    row[cursor+p] = 1.; cursor += 4
    row[cursor+(3 if trump == -1 else trump)] = 1.; cursor += 4
    row[cursor:cursor+3] = [n/3, S/3, R/4]; cursor += 3
    row[cursor+np.array(legal)] = 1.; cursor += 3
    assert cursor == WIDTH
    return row
