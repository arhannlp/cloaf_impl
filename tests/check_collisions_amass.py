"""
How often does each collision test fire on CLEAN mocap (AMASS)?
Clean motion should rarely collide, so a good test gives a low Col.Rate here.

  python tests/check_collisions_amass.py --smpl data/body_models/smpl --amass data/amass_smpl_30fps.npz

Prints, for each configuration, Col.Rate@0 / Col.Rate@100 over random frames and,
for the "parts" test, which body parts the colliding vertices belong to.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloaf.body import SMPLBody  # noqa: E402
from cloaf.collision import PART_NAMES, collision_rate  # noqa: E402

CONFIGS = [
    ("full, rest-pose ignored", dict(collision_mode="full", ignore_excluded_regions=False)),
    ("full, + crotch/armpit ignored", dict(collision_mode="full", ignore_excluded_regions=True)),
    ("parts, rest-pose ignored", dict(collision_mode="parts", ignore_excluded_regions=False)),
    ("parts, + crotch/armpit ignored (default)", dict(collision_mode="parts", ignore_excluded_regions=True)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl", default="data/body_models/smpl")
    ap.add_argument("--amass", default="data/amass_smpl_30fps.npz")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    P = np.load(a.amass)["body_pose"]
    rng = np.random.RandomState(0)
    pick = rng.choice(len(P), size=min(a.frames, len(P)), replace=False)
    poses = torch.as_tensor(P[pick])
    print(f"{len(pick)} random frames from {a.amass}\n")
    print(f"{'configuration':42s} {'Col@0':>7s} {'Col@100':>8s} {'mean':>7s} {'s/frame':>8s}")
    for name, kw in CONFIGS:
        body = SMPLBody(a.smpl, device=a.device, **kw)
        t = time.time()
        counts, parts = [], torch.zeros(len(PART_NAMES))
        for i in range(0, len(poses), 16):
            v = body.vertices(poses[i:i + 16].to(body.device))
            for vi in v:
                m = body.inside(vi)
                counts.append(int(m.sum()))
                if body.collision_mode == "parts" and m.any():
                    parts += torch.bincount(body._part_test.vertex_part[m], minlength=len(PART_NAMES)).cpu()
        dt = (time.time() - t) / len(poses)
        c = torch.tensor(counts)
        print(f"{name:42s} {collision_rate(c, 0):6.1f}% {collision_rate(c, 100):7.1f}% "
              f"{c.float().mean():7.1f} {dt:8.2f}")
        if parts.sum() > 0:
            top = sorted(((int(n), PART_NAMES[k]) for k, n in enumerate(parts) if n > 0), reverse=True)[:6]
            print(f"{'':42s} colliding vertices by part: " + ", ".join(f"{p} {n}" for n, p in top))
    print("\nClean mocap should give a low Col.Rate. For reference, the paper reports"
          " 39.2 % (@0) for HMR2.0 predictions on 3DPW, which are noisier than mocap.")


if __name__ == "__main__":
    main()
