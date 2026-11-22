"""
STEP 2 — INTEGRATION TEST WITHOUT TRAINING (Sec. 4.4, Fig. 3-4, Supp. C)
A simple field that moves the left hand towards a target point:
  f(x) = F (x_T - x)/(||x_T - x|| + eps), F=1e-3, eps=1e-6, r_in=10mm, r_out=30mm
This shows that the ODE + Jacobian + pseudo-inverse chain works before any training.

  python scripts/demo_target_field.py --smpl data/body_models/smpl [--box]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloaf.body import SMPLBody, L_SHOULDER  # noqa: E402
from cloaf.integrate import CLOAFODE, TargetField  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--T", type=float, default=400.0, help="integration time (F=1e-3 m per unit)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--box", action="store_true", help="put a box obstacle in front of the target")
    ap.add_argument("--out", default="results/target_field.npz")
    a = ap.parse_args()

    body = SMPLBody(a.smpl, device=a.device)
    th0 = torch.zeros(1, 69, device=body.device, dtype=body.dtype)
    v0 = body.vertices(th0)[0]
    # source region: left hand/wrist (vertices whose dominant skinning weight is L_wrist/L_hand)
    W = body.smpl.lbs_weights
    region = (W.argmax(-1) == 20) | (W.argmax(-1) == 22)   # 20=L_wrist, 22=L_hand
    target = v0[region].mean(0) + torch.tensor([0.0, 0.45, 0.15], device=body.device, dtype=body.dtype)

    box = None
    if a.box:
        c = 0.5 * (v0[region].mean(0) + target)
        box = (c - 0.08, c + 0.08)
    field = TargetField(body, region, target.float(), box=box)
    sample = body.sample_surface(1000)
    ode = CLOAFODE(body, field, th0, sample, t1=a.T)     # theta_1 is unused (the field has its own target)
    ts = torch.linspace(0, a.T, a.steps, device=body.device, dtype=body.dtype)
    with torch.no_grad():
        traj = odeint(ode, th0, ts, method="rk4", options=dict(step_size=a.T / 200))
        verts = body.vertices(traj[:, 0])
        hand = verts[:, region].mean(1)
        print("hand-target distance (m):", (hand - target).norm(dim=-1)[::5].cpu().numpy().round(3))
        print("colliding vertices:", body.count_collisions(verts)[::5].cpu().numpy())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, verts=verts.float().cpu().numpy(), faces=body.faces.cpu().numpy(),
             target=target.cpu().numpy())


if __name__ == "__main__":
    main()
