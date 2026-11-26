"""
Figures for Sec. 4.4 / Fig. 3-4 / Supp. C: a simple hand-crafted field moves the left hand to a
target point, without and with a box obstacle (field = 0 inside the box). No network involved.

  python scripts/fig_target_field.py --smpl data/body_models/smpl --out results/fig34
  -> results/fig34.png         snapshots of the body (top: no obstacle, bottom: box)
     results/fig34_curves.png  hand-target distance and colliding vertices over time
     results/fig34.npz         the trajectories (vertices), for GIFs

Runs on CPU in a few minutes (--substeps lower = faster, less accurate).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchdiffeq import odeint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloaf.body import SMPLBody  # noqa: E402
from cloaf.integrate import CLOAFODE, TargetField  # noqa: E402


def run(body, region, target, box, T, steps, substeps, S):
    th0 = torch.zeros(1, 69, device=body.device, dtype=body.dtype)
    field = TargetField(body, region, target.float(), box=box)
    ode = CLOAFODE(body, field, th0, body.sample_surface(S), t1=T)
    ts = torch.linspace(0, T, steps, device=body.device, dtype=body.dtype)
    with torch.no_grad():
        traj = odeint(ode, th0, ts, method="rk4", options=dict(step_size=T / substeps))
        verts = body.vertices(traj[:, 0])
        dist = (verts[:, region].mean(1) - target).norm(dim=-1)
        coll = body.count_collisions(verts)
    return verts.cpu().numpy(), dist.cpu().numpy(), coll.cpu().numpy()


def box_edges(lo, hi):
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return [(c[i], c[j]) for i in range(8) for j in range(i + 1, 8) if (np.abs(c[i] - c[j]) > 1e-9).sum() == 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl", required=True)
    ap.add_argument("--out", default="results/fig34")
    ap.add_argument("--T", type=float, default=400.0, help="integration time (F = 1e-3 m per unit)")
    ap.add_argument("--steps", type=int, default=41, help="saved time steps")
    ap.add_argument("--substeps", type=int, default=100, help="RK4 steps")
    ap.add_argument("--S", type=int, default=1000)
    ap.add_argument("--azim", type=float, default=35)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    body = SMPLBody(a.smpl, device=a.device)
    v0 = body.vertices(torch.zeros(1, 69, device=body.device, dtype=body.dtype))[0]
    W = body.smpl.lbs_weights
    region = (W.argmax(-1) == 20) | (W.argmax(-1) == 22)            # left wrist + left hand
    target = v0[region].mean(0) + torch.tensor([0.0, 0.45, 0.15], device=body.device, dtype=body.dtype)
    c = 0.5 * (v0[region].mean(0) + target)
    box = (c - 0.08, c + 0.08)

    res = {}
    for name, b in [("free", None), ("box", box)]:
        print(f"integrating ({name}) ...", flush=True)
        res[name] = run(body, region, target, b, a.T, a.steps, a.substeps, a.S)
        v, d, k = res[name]
        print(f"  hand-target distance {d[0] * 100:.1f} -> {d[-1] * 100:.1f} cm, "
              f"max colliding vertices {int(k.max())}", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f"{out}.npz", faces=body.faces.cpu().numpy(), target=target.cpu().numpy(),
             box_lo=box[0].cpu().numpy(), box_hi=box[1].cpu().numpy(),
             **{f"{n}_{k}": x for n, r in res.items() for k, x in zip(["verts", "dist", "coll"], r)})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    zup = lambda p: np.asarray(p)[..., [0, 2, 1]]                      # SMPL y-up -> z-up
    faces = body.faces.cpu().numpy()
    tg = zup(target.cpu().numpy())
    lo, hi = zup(box[0].cpu().numpy()), zup(box[1].cpu().numpy())
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    light = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    idx = np.linspace(0, a.steps - 1, 4).round().astype(int)
    center = zup(v0.cpu().numpy()).mean(0)
    reg = region.cpu().numpy()

    fig = plt.figure(figsize=(12, 6.6))
    for row, name in enumerate(["free", "box"]):
        v, d, k = res[name]
        hand = zup(v[:, reg].mean(1))
        for col, t in enumerate(idx):
            ax = fig.add_subplot(2, 4, 4 * row + col + 1, projection="3d")
            tri = zup(v[t])[faces]
            n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            shade = (0.35 + 0.65 * np.abs(n @ light))[:, None] * np.array([0.72, 0.80, 0.95])
            ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(shade, 0, 1), edgecolors="none"))
            ax.plot(*hand[:t + 1].T, color="tab:orange", lw=1.5)
            ax.scatter(*tg, color="red", s=30, depthshade=False)
            if name == "box":
                for p, q in box_edges(lo, hi):
                    ax.plot(*np.stack([p, q]).T, color="k", lw=0.8)
            r = 0.75
            ax.set_xlim(center[0] - r, center[0] + r)
            ax.set_ylim(center[1] - r, center[1] + r)
            ax.set_zlim(center[2] - r, center[2] + r)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=10, azim=a.azim)
            ax.axis("off")
            ax.set_title(f"t = {t / (a.steps - 1):.2f} T   d = {d[t] * 100:.1f} cm", fontsize=9)
        fig.text(0.01, 0.75 - 0.48 * row, "no obstacle" if name == "free" else "box obstacle",
                 rotation=90, va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    tt = np.linspace(0, 1, a.steps)
    for name, ls in [("free", "-"), ("box", "--")]:
        v, d, k = res[name]
        axes[0].plot(tt, d * 100, ls, label="no obstacle" if name == "free" else "box obstacle")
        axes[1].plot(tt, k, ls)
    axes[0].set_ylabel("hand-target distance (cm)")
    axes[1].set_ylabel("colliding vertices")
    for ax in axes:
        ax.set_xlabel("t / T")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(f"{out}_curves.png", dpi=150)
    print(f"saved {out}.png, {out}_curves.png, {out}.npz")


if __name__ == "__main__":
    main()
