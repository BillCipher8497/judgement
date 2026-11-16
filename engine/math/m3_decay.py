"""M3 decay decomposition: C(n) = P(f=b*) * E[gap|f=b*], and a discretised-Gaussian
prediction of E[gap|f=b*] from the dealer's mean trick s.d. sigma(n):
  P(t) ∝ exp(-(t-mu)^2/(2 sigma^2)) on integers, mu = b* + delta, delta ~ U(-1/2, 1/2),
  gap = EV(b*) - max(EV(b*-1), EV(b*+1)), EV(b) = (10+b) P(b), b* ~ n/4 (dealer mean bid level).
Reads math/results/m3_hook.json, writes math/results/m3_decay.json."""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results"


def gauss_gap(sigma, b, H=10):
    gs = []
    for d in np.linspace(-0.5, 0.5, 201):
        t = np.arange(-30, 31)
        w = np.exp(-(t - d) ** 2 / (2 * sigma ** 2))
        P = w / w.sum()
        i = 30
        ev = lambda k: (H + b + k) * P[i + k]  # noqa: E731
        if ev(0) < max(ev(-1), ev(1)):
            continue  # b* not at the centre for this delta
        gs.append(ev(0) - max(ev(-1) if b > 0 else -1e9, ev(1)))
    return float(np.mean(gs))


def main():
    d = json.load(open(R / "m3_hook.json"))
    out = {}
    for n, v in d["by_n"].items():
        n = int(n)
        sig = v["mean_sd_dealer"]
        g = gauss_gap(sig, max(round(n / 4), 0))
        out[n] = {"P(f=b*)": v["P(f==b*)"][0], "gap|hit_model": v["E[gap|f==b*]"][0], "gap_gauss": g,
                  "sigma": sig, "C_pred": v["C_pred"][0], "C_gauss": v["P(f==b*)"][0] * g,
                  "measured": v["measured_crn"][0], "measured_ci": v["measured_crn"][1],
                  "ratio_meas_over_pred": v["measured_crn"][0] / v["C_pred"][0]}
    json.dump(out, open(R / "m3_decay.json", "w"), indent=1)
    L = ["n | P(f=b*) | E[gap|f=b*] model | gap Gauss(sigma) | sigma | C_pred | C_gauss | measured E5 (CRN) | meas/pred"]
    for n, v in out.items():
        L.append(f"{n} | {v['P(f=b*)']:.3f} | {v['gap|hit_model']:.2f} | {v['gap_gauss']:.2f} | {v['sigma']:.2f} | "
                 f"{v['C_pred']:.2f} | {v['C_gauss']:.2f} | {v['measured']:.2f}±{v['measured_ci']:.2f} | {v['ratio_meas_over_pred']:.2f}")
    open(R / "m3_tables.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
