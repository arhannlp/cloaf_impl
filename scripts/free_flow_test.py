"""
Where does the non-intersection guarantee break? Free flow vs projected flow.

For a few (Θ0, Θ1) pairs taken from a sequence with an injected collision
(Θ0: last collision-free frame before the collision, Θ1: a colliding frame),
the same field is integrated in two ways:

  free flow      : every one of the 6890 vertices moves with the field directly,
                   no SMPL. If the field is a function of position, trajectories
                   cannot cross, so NO new collision should appear (paper, Eq. 1).
  projected flow : CLOAF as implemented: the velocities are projected onto the
                   SMPL parameters with J^+ (Eq. 6).

and the projection error ||J J^+ f - f|| / ||f|| (the share of the requested
motion SMPL cannot follow) is reported.

Reading the result:
  free flow clean, projected flow colliding -> the projection breaks the guarantee
  free flow colliding too                   -> the field or the collision test is the issue

  python scripts/free_flow_test.py --smpl data/body_models/smpl \
      --seq data/amass/SFU/0005/0005_2FeetJump001_poses.npz --field smooth_target
  (--field also accepts point_target or a checkpoint path)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cloaf.body import SMPLBody  # noqa: E402
from cloaf.integrate import CLOAFODE, PointTargetField, SmoothTargetField  # noqa: E402
from cloaf.model import DT_MIN, MotionField  # noqa: E402
from run_on_sequence import corrupt, load_sequence  # noqa: E402


def free_velocity(kind, net, sigma, idx):
    """Velocity of every vertex for the free flow. Same field types as CLOAF."""
    def vel(t, X, X1, th_t, th1):
        dt = torch.full((1,), max(1.0 - t, DT_MIN), device=X.device, dtype=X.dtype)
        if kind == "point_target":
            return (X1 - X) / dt                                   # per vertex: no guarantee
        if kind == "smooth_target":
            c, vc = X[:, idx], (X1[:, idx] - X[:, idx]) / dt        # kernel centres = sampled vertices
            K = torch.softmax(-torch.cdist(X, c) ** 2 / (2 * sigma ** 2), dim=-1)
            return K @ vc
        return net(X.float(), th_t.float(), th1.float(), dt.float(), x_target=X1.float()).to(X)
    return vel


def rk4(vel, X0, X1, th0, th1, steps, check, body):
    """Integrate the free flow; count collisions every `check` steps."""
    X, h, worst = X0.clone(), 1.0 / steps, 0
    for i in range(steps):
        t = i * h
        th = lambda s: th0 + s * (th1 - th0)                     # Θ~(t) to condition the network
        k1 = vel(t, X, X1, th(t), th1)
        k2 = vel(t + h / 2, X + h / 2 * k1, X1, th(t + h / 2), th1)
        k3 = vel(t + h / 2, X + h / 2 * k2, X1, th(t + h / 2), th1)
        k4 = vel(t + h, X + h * k3, X1, th(t + h), th1)
        X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if (i + 1) % check == 0 or i == steps - 1:
            worst = max(worst, int(body.inside(X[0]).sum()))
    return X, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--field", default="smooth_target",
                    help="smooth_target, point_target, or a checkpoint path")
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--corrupt_start", type=int, default=30)
    ap.add_argument("--corrupt_len", type=int, default=30)
    ap.add_argument("--corrupt_amp", type=float, default=1.2)
    ap.add_argument("--offsets", default="5,15,25", help="colliding frames tested, from corrupt_start")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--S", type=int, default=1000)
    ap.add_argument("--collision_min", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    body = SMPLBody(a.smpl, device=a.device, collision_min=a.collision_min)
    bp, go = load_sequence(a.seq, 30, 0, a.frames)
    clean = torch.as_tensor(bp, device=body.device, dtype=body.dtype)
    go = torch.as_tensor(go, device=body.device, dtype=body.dtype)
    inp = corrupt(body, clean, go, a.corrupt_start, a.corrupt_len, a.corrupt_amp)

    kind = a.field if a.field in ("smooth_target", "point_target") else "network"
    net = None
    if kind == "network":
        net = MotionField().to(body.device).eval()
        net.load_state_dict(torch.load(a.field, map_location=body.device))
        proj_field = net
    else:
        proj_field = SmoothTargetField(a.sigma) if kind == "smooth_target" else PointTargetField()
    g = torch.Generator(device=body.device).manual_seed(0)
    n_verts = body.smpl.v_template.shape[0]
    idx = torch.randperm(n_verts, device=body.device, generator=g)[:min(a.S, n_verts)]
    vel = free_velocity(kind, net, a.sigma, idx)

    with torch.no_grad():
        counts = body.count_collisions(body.vertices(inp, None, go))
    free_frames = [k for k in range(a.corrupt_start) if counts[k] <= a.collision_min]
    if not free_frames:
        raise SystemExit("no collision-free frame before the injected collision; try another sequence")
    k0 = free_frames[-1]
    print(f"field: {a.field}" + (f" (sigma {a.sigma} m)" if kind == "smooth_target" else ""))
    print(f"start pose: frame {k0} ({int(counts[k0])} colliding vertices)\n")
    print(f"{'target':>7s} {'Θ1 coll.':>9s} | {'free: coll. end/worst':>21s} {'dist to Θ1':>11s} | "
          f"{'projected: coll. end/worst':>26s} {'dist to Θ1':>11s} {'proj. error':>12s}")

    rows = []
    for off in [int(o) for o in a.offsets.split(",")]:
        k = a.corrupt_start + off
        g_k = go[k:k + 1]                                  # same global orientation for both poses
        th0, th1 = inp[k0:k0 + 1], inp[k:k + 1]
        with torch.no_grad():
            X0, X1 = body.vertices(th0, None, g_k), body.vertices(th1, None, g_k)
            c1 = int(body.inside(X1[0]).sum())
            Xf, worst_f = rk4(vel, X0, X1, th0, th1, a.steps, max(1, a.steps // 10), body)
            cf = int(body.inside(Xf[0]).sum())
            df = 1000 * (Xf - X1).norm(dim=-1).mean().item()

            sample = body.sample_surface(a.S, generator=torch.Generator(device=body.device).manual_seed(0))
            ode = CLOAFODE(body, proj_field, th1, sample, None, g_k)
            ts = torch.linspace(0, 1, 11, device=body.device, dtype=body.dtype)
            traj = odeint(ode, th0, ts, method="rk4", options=dict(step_size=1.0 / a.steps))
            cp_all = [int(body.inside(body.vertices(traj[i], None, g_k)[0]).sum()) for i in range(len(ts))]
            Xp = body.vertices(traj[-1], None, g_k)
            dp = 1000 * (Xp - X1).norm(dim=-1).mean().item()
            pe = float(np.mean(ode.proj_err))
        rows.append((k, c1, cf, worst_f, df, cp_all[-1], max(cp_all), dp, pe))
        print(f"{k:7d} {c1:9d} | {cf:10d} / {worst_f:<8d} {df:9.1f}mm | "
              f"{cp_all[-1]:14d} / {max(cp_all):<9d} {dp:9.1f}mm {100 * pe:10.1f} %")

    print("\nColliding vertices: 'end' at t = 1, 'worst' = maximum along the path.")
    print("Projection error: share of the requested point motion that SMPL cannot follow (mean over steps).")


if __name__ == "__main__":
    main()
