"""
Which body parts collide in the HMR2.0 predictions? (calibration of the collision test, Col.Rate)

For one track of scripts/hmr2_3dpw.py output it prints, per frame, the number of colliding
vertices with the predicted shape and with the mean shape (betas = 0), the part pairs that
contribute (vertex of part A inside part B, summed over the frames), and saves a front/side
picture of one frame with the colliding vertices in red.

  python scripts/diagnose_collisions.py --smpl data/body_models/smpl --pred results/hmr2_quick.npz \
      --key downtown_arguing_00/0 --png results/diag.png
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloaf.body import SMPLBody  # noqa: E402
from cloaf.collision import PART_NAMES, _above  # noqa: E402


def pair_counts(body, v):
    """Counter {(part of the vertex, part it is inside): n} for one frame (V,3)."""
    t = body._part_test
    faces = body.faces
    face_parts = t.vertex_part[faces]
    out = Counter()
    hit = body.inside(v)
    for p in range(len(PART_NAMES)):
        idx = ((t.vertex_part == p) & hit).nonzero().squeeze(-1)
        if not len(idx):
            continue
        for q in range(len(PART_NAMES)):
            if t.near[p, q]:
                continue
            fq = faces[(face_parts == q).all(-1)]
            if len(fq):
                n = int(_above(v, fq, v[idx], t.thresh).sum())
                if n:
                    out[(PART_NAMES[p], PART_NAMES[q])] += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--key", default=None, help="track, e.g. downtown_arguing_00/0 (default: first)")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--png", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    pred = {k: v.item() for k, v in np.load(a.pred, allow_pickle=True).items() if k != "summary"}
    key = a.key or sorted(pred)[0]
    P = pred[key]
    n = min(a.frames, len(P["frames"]))
    body = SMPLBody(a.smpl, device=a.device)
    f64 = torch.float64
    bp = torch.as_tensor(P["body_pose"][:n], device=body.device, dtype=f64)
    be = torch.as_tensor(P["betas"][:n], device=body.device, dtype=f64)
    with torch.no_grad():
        v = body.vertices(bp, be)
        v0 = body.vertices(bp)
        c, c0 = body.count_collisions(v).cpu(), body.count_collisions(v0).cpu()
        rest = body.count_collisions(body.vertices(torch.zeros_like(bp[:1]), be[:1])).item()
    print(f"{key}: betas (first frame) {np.round(P['betas'][0], 2).tolist()}")
    print(f"rest pose with these betas: {rest} colliding vertices")
    print("frame  colliding(pred betas)  colliding(betas=0)")
    for i in range(n):
        print(f"{int(P['frames'][i]):5d}  {int(c[i]):8d}  {int(c0[i]):8d}")

    tot = Counter()
    for i in range(n):
        tot += pair_counts(body, v[i])
    print("\nvertex of part A inside part B, summed over the frames:")
    for (pa, pb), k in tot.most_common(12):
        print(f"  {pa:12s} in {pb:12s} {k:7d}  ({k / n:.0f} per frame)")

    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        i = int(c.argmax())
        with torch.no_grad():
            hit = body.inside(v[i]).cpu().numpy()
        x = v[i].cpu().numpy()
        x = x - x.mean(0)
        fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        for ax, (h, name) in zip(axes, [(0, "front (x-y)"), (2, "side (z-y)")]):
            ax.scatter(x[~hit, h], x[~hit, 1], s=0.3, c="0.7")
            ax.scatter(x[hit, h], x[hit, 1], s=2, c="red")
            ax.set_aspect("equal")
            ax.set_title(name)
            ax.axis("off")
        fig.suptitle(f"{key} frame {int(P['frames'][i])}: {int(c[i])} colliding vertices (red)")
        Path(a.png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.png, dpi=120, bbox_inches="tight")
        print(f"saved {a.png}")


if __name__ == "__main__":
    main()
