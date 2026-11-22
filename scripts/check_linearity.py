"""
STEP 1 — FIRST NUMERICAL CHECK, NO TRAINING NEEDED
Reproduces Fig. 5 and Supp. Fig. B.7 / B.8 of the paper.

  f(δΘ) = X(Θ0 + δΘ) - X(Θ0)          (Eq. B.13)
  δΘ^   = J^† f
  RE    = ||δΘ^ - δΘ|| / ||δΘ||        (Eq. 12)

Target values from the paper (present them to the supervisor as a comparison table):
  - ||δΘ|| = 1e-2, S = 6890        -> RE ≈ 7.1e-2   (Fig. B.8 right)
  - ||δΘ|| = 1e-1, S = 6890        -> RE ≈ 7.0e-1   (Fig. B.8 left)
  - ||δΘ|| ∈ [1e-7, 1e-1] (green)   -> RE ≲ 1e-1    (Fig. B.7)
  - S = 10 -> RE ≈ 0.7 ; S ≥ 100 -> RE ≈ 0.05-0.08 ; S=1000 good trade-off (Fig. 5)
  - Jacobian time: ~15-20 ms for S ≤ 1000, ~320 ms for S = 6890 (A100)
  - Averaged over 10 restarts, double precision.

Usage:
  python scripts/check_linearity.py --smpl data/body_models/smpl [--amass data/amass_smpl_30fps.npz]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloaf.body import SMPLBody, pinv_solve  # noqa: E402


def rel_error(body, th0, mag, S, gen, jac_method):
    sample = None if S >= 6890 else body.sample_surface(S, generator=gen)
    d = torch.randn(th0.shape, device=body.device, dtype=body.dtype, generator=gen)
    d = d / d.norm() * mag
    with torch.no_grad():
        f = body.points(th0 + d, sample) - body.points(th0, sample)
        if body.device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        J = body.jacobian(th0, sample, method=jac_method)
        if body.device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter() - t
        d_hat = pinv_solve(J, f.reshape(1, -1))
    return ((d_hat - d).norm() / d.norm()).item(), f.norm(dim=-1).mean().item(), t * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl", required=True)
    ap.add_argument("--amass", default=None, help="sample Θ0 from real poses")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--jac", default="fd", choices=["fd", "jacfwd"])
    ap.add_argument("--out", default="results/linearity.npz")
    a = ap.parse_args()

    body = SMPLBody(a.smpl, device=a.device)
    gen = torch.Generator(device=body.device).manual_seed(0)
    if a.amass:
        P = torch.as_tensor(np.load(a.amass)["body_pose"], dtype=body.dtype, device=body.device)
        pick = lambda: P[torch.randint(len(P), (1,), device=body.device, generator=gen)]
    else:
        # TODO: the paper only says "we sample an SMPL vector" -> distribution unknown
        pick = lambda: 0.2 * torch.randn(1, 69, device=body.device, dtype=body.dtype, generator=gen)

    # --- Fig. B.7: S=6890, sweep over the magnitude ---
    mags = np.logspace(-9, 1, 21)
    print("\n[Fig. B.7]  S=6890")
    print(f"{'||dTheta||':>12} {'||f||':>12} {'RE':>10}")
    b7 = []
    for m in mags:
        r = np.mean([rel_error(body, pick(), m, 6890, gen, a.jac)[:2] for _ in range(a.restarts)], 0)
        b7.append(r)
        print(f"{m:12.1e} {r[1]:12.3e} {r[0]:10.3e}")

    # --- Fig. 5: ||δΘ||=1e-2, sweep over S ---
    Ss = [10, 30, 100, 300, 1000, 2000, 4000, 6890]
    print("\n[Fig. 5]  ||dTheta||=1e-2")
    print(f"{'S':>6} {'RE':>10} {'time ms':>10}")
    f5 = []
    for S in Ss:
        r = np.mean([rel_error(body, pick(), 1e-2, S, gen, a.jac) for _ in range(a.restarts)], 0)
        f5.append(r)
        print(f"{S:6d} {r[0]:10.3e} {r[2]:10.1f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, mags=mags, b7=np.array(b7), S=np.array(Ss), f5=np.array(f5))


if __name__ == "__main__":
    main()
