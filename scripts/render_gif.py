"""
Render a correction result as an animated GIF: input (left) and CLOAF output (right),
faces touching a colliding vertex in red. Works headless (Colab), no aitviewer needed.

  python scripts/render_gif.py --smpl data/body_models/smpl --result results/x.npz --out results/x.gif
  (in Colab:  from IPython.display import Image; Image('results/x.gif'))

Accepts the output of cloaf_correct.py and of run_on_sequence.py ('input', 'corrected').
The body is shown in its own frame (global rotation and translation removed), front view,
so the pose changes are easy to see (--azim 90 front, -90 back, 0 side).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloaf.body import SMPLBody  # noqa: E402


def shade(tri, base, red, colliding_faces):
    """Flat shading from the face normals, red for colliding faces."""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.3, 0.5, 0.8])
    light /= np.linalg.norm(light)
    k = 0.35 + 0.65 * np.abs(n @ light)
    col = np.where(colliding_faces[:, None], red, base) * k[:, None]
    return np.clip(col, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", default=None, help="default: result path with .gif")
    ap.add_argument("--every", type=int, default=2, help="render every n-th frame")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--size", type=float, default=4.0, help="inches per panel")
    ap.add_argument("--azim", type=float, default=90, help="camera azimuth: 90 = front, -90 = back")
    ap.add_argument("--elev", type=float, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    r = np.load(a.result)
    body = SMPLBody(a.smpl, device=a.device)
    betas = torch.as_tensor(r["betas"], device=body.device, dtype=body.dtype) if "betas" in r else None
    frames = list(range(a.start, a.end or len(r["input"]), a.every))
    faces = body.faces.cpu().numpy()

    def mesh(key, k):
        th = torch.as_tensor(r[key][k:k + 1], device=body.device, dtype=body.dtype)
        with torch.no_grad():
            v = body.vertices(th, None if betas is None else betas[k:k + 1])[0]
            inside = body.inside(v).cpu().numpy()
        v = v.cpu().numpy()
        v = v - v.mean(0)
        return v[:, [0, 2, 1]], inside          # SMPL y-up -> matplotlib z-up

    fig = plt.figure(figsize=(2 * a.size, a.size * 1.1))
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    base, red = np.array([0.72, 0.80, 0.95]), np.array([0.95, 0.15, 0.12])

    def draw(i):
        k = frames[i]
        for ax, key, name in zip(axes, ["input", "corrected"], ["input", "CLOAF"]):
            ax.cla()
            v, inside = mesh(key, k)
            tri = v[faces]
            col = shade(tri, base, red, inside[faces].any(-1))
            ax.add_collection3d(Poly3DCollection(tri, facecolors=col, edgecolors="none"))
            ax.set_xlim(-0.9, 0.9)
            ax.set_ylim(-0.9, 0.9)
            ax.set_zlim(-0.9, 0.9)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=a.elev, azim=a.azim)
            ax.axis("off")
            ax.set_title(f"{name}  frame {k}  colliding vertices {int(inside.sum())}", fontsize=9)
        print(f"\rframe {k}", end="", flush=True)

    anim = FuncAnimation(fig, draw, frames=len(frames))
    out = a.out or str(Path(a.result).with_suffix(".gif"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=max(1, 30 // a.every)))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
