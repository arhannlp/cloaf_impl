"""
Watch the output of run_on_sequence.py side by side in aitviewer
(run this on a machine with a screen):

  left : input (colliding vertices in red)
  right: CLOAF output (colliding vertices in red)

  python scripts/view_results.py --smpl data/body_models/smpl --result results/sequence.npz

Space = play/pause. The GUI can also export a video.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloaf.body import SMPLBody  # noqa: E402


def colored(body, poses, go, name, offset):
    from aitviewer.renderables.meshes import Meshes
    with torch.no_grad():
        v = body.vertices(torch.as_tensor(poses, dtype=body.dtype, device=body.device), None,
                          torch.as_tensor(go, dtype=body.dtype, device=body.device))
    colors = np.tile([0.7, 0.8, 1.0, 1.0], (v.shape[0], v.shape[1], 1))
    for i in range(v.shape[0]):
        colors[i, body.inside(v[i]).cpu().numpy()] = [1.0, 0.1, 0.1, 1.0]
    return Meshes(v.float().cpu().numpy(), body.faces.cpu().numpy(), vertex_colors=colors,
                  name=name, position=np.array(offset))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl", required=True)
    ap.add_argument("--result", default="results/sequence.npz")
    a = ap.parse_args()

    from aitviewer.viewer import Viewer
    body = SMPLBody(a.smpl, device="cpu")
    r = np.load(a.result)
    viewer = Viewer()
    viewer.scene.add(colored(body, r["input"], r["global_orient"], "Input", [-0.6, 0, 0]),
                     colored(body, r["corrected"], r["global_orient"], "CLOAF", [0.6, 0, 0]))
    viewer.run()


if __name__ == "__main__":
    main()
