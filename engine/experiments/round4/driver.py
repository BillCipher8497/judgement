"""Round-4 job driver: 2 worker processes over an interleaved chunk list (so a deadline cut stays balanced).
python driver.py DEADLINE_EPOCH   -> results/f_<tag>.npz per chunk; log results/driver.log"""
import sys, time, subprocess
from multiprocessing import Pool
from pathlib import Path

H = Path(__file__).parent
G1_BASE, G2_BASE = 500_000_000, 600_000_000
G1_POOLS = {"a": "hyb_noisy:k=16,c=5,p=0.3", "b": "hyb_honour:k=16,c=5"}
G1_ARMS = {"ref": None, "plain": "hybrid:k=16,c=5", "bi": "hybrid_bi:k=16,c=5", "bit": "hybrid_bi:k=16,c=5,eps=0.1"}
G2_POOL = "hybrid_bi:k=32,c=6"
G2_ARMS = {"ref": G2_POOL, "plain": "hybrid:k=32,c=6", "c7": "hybrid_bi:k=32,c=7"}


def jobs():
    J = []
    for i in range(4):                      # G1: 4 chunks x 125 games
        for p, pool in G1_POOLS.items():
            for a, ch in G1_ARMS.items():
                J.append((pool, ch or pool, 125 * i, 125, f"g1{p}_{a}_{i}", G1_BASE))
    for i in range(10):                     # G2: 10 chunks x 20 games
        for a, ch in G2_ARMS.items():
            J.append((G2_POOL, ch, 20 * i, 20, f"g2_{a}_{i}", G2_BASE))
    return J


def _init(d):
    global DL
    DL = d


def work(j):
    if time.time() > DL:
        return
    with open(H / "results" / "driver.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} start {j[4]}\n"); f.flush()
        subprocess.run([sys.executable, "run.py", *map(str, j), str(DL)], cwd=H, stdout=f, stderr=f)


if __name__ == "__main__":
    DL = float(sys.argv[1])
    with Pool(2, initializer=_init, initargs=(DL,)) as P:
        list(P.imap(work, jobs(), chunksize=1))
