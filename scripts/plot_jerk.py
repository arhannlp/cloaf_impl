"""
Per-frame view of one or more run_on_sequence.py results: where do the jumps happen?

Three panels over the frames:
  1. jerk (mean per-vertex third difference, mm): sudden jumps show up as peaks
  2. distance to the clean motion (mm)
  3. colliding vertices after correction (and before, from the first file)

  python scripts/plot_jerk.py --smpl data/body_models/smpl \
      --results results/jump_smooth_c0.05_colliding_only.npz results/jump_smooth_c0.05_all.npz \
      --window 20 39 --out results/jerk.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloaf.body import SMPLBody  # noqa: E402


def verts(body, poses, go):
    with torch.no_grad():
        return body.vertices(torch.as_tensor(poses, device=body.device, dtype=body.dtype), None,
                             torch.as_tensor(go, device=body.device, dtype=body.dtype))


def jerk_per_frame(v):
    """(N,V,3) -> (N,) mean per-vertex third difference in mm; first 3 frames are NaN."""
    j = 1000 * (v[3:] - 3 * v[2:-1] + 3 * v[1:-2] - v[:-3]).norm(dim=-1).mean(-1)
    return np.concatenate([np.full(3, np.nan), j.cpu().numpy()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl", required=True)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--window", nargs=2, type=int, default=None,
                    help="first and last frame of the injected collision (shaded)")
    ap.add_argument("--out", default="results/jerk.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    body = SMPLBody(a.smpl, device=a.device)
    labels = a.labels or [Path(f).stem for f in a.results]
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    first = np.load(a.results[0])
    vc, vi = verts(body, first["clean"], first["global_orient"]), verts(body, first["input"], first["global_orient"])
    frames = np.arange(len(first["clean"]))
    ax[0].plot(frames, jerk_per_frame(vc), color="0.6", lw=1, label="clean motion")
    ax[0].plot(frames, jerk_per_frame(vi), color="0.2", lw=1, ls="--", label="input")
    ax[1].plot(frames, 1000 * (vi - vc).norm(dim=-1).mean(-1).cpu().numpy(), color="0.2", ls="--", label="input")
    ax[2].plot(frames, first["before"], color="0.2", ls="--", label="input")

    print(f"{'result':45s} {'max jerk (frame)':>18s} {'mean jerk in / out of window':>30s}")
    for f, lab in zip(a.results, labels):
        r = np.load(f)
        vo = verts(body, r["corrected"], r["global_orient"])
        j = jerk_per_frame(vo)
        ax[0].plot(frames, j, lw=1.5, label=lab)
        ax[1].plot(frames, 1000 * (vo - vc).norm(dim=-1).mean(-1).cpu().numpy(), lw=1.5, label=lab)
        ax[2].plot(frames, r["after"], lw=1.5, label=lab)
        k = int(np.nanargmax(j))
        if a.window:
            w = (frames >= a.window[0]) & (frames <= a.window[1])
            inside, outside = np.nanmean(j[w]), np.nanmean(j[~w])
            extra = f"{inside:12.1f} / {outside:<12.1f}"
        else:
            extra = ""
        print(f"{lab:45s} {np.nanmax(j):10.1f} ({k:3d})   {extra}")

    for x, title in zip(ax, ["jerk (mm)", "distance to clean motion (mm)", "colliding vertices"]):
        if a.window:
            x.axvspan(a.window[0], a.window[1], color="tab:red", alpha=0.08)
        x.set_ylabel(title)
        x.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[2].set_xlabel("frame")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
