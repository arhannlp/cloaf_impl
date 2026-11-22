"""
STEP 3/5 — 3DPW-test evaluation (Table 1, 2, 4).

Input: HMR2.0 (4D-Humans) outputs, SMPL parameters per frame.
  predictions.npz contains, for each sequence:
    <seq>/<person>/body_pose (N,69) axis-angle, global_orient (N,3), betas (N,10)
  (4D-Humans outputs rotation matrices -> convert to axis-angle with
   pytorch3d.transforms.matrix_to_axis_angle or the roma library)

First measure ONLY the baseline with --no_cloaf; match the HMR2.0 row:
   MPJPE 82.0 | P-MPJPE 52.7 | Accel 16.1 | Col.Rate@0 39.2 | @100 21.6
Then with CLOAF:
   MPJPE 82.3 | P-MPJPE 53.2 | Accel 9.4  | Col.Rate@0 0.0

  python scripts/eval_3dpw.py --smpl data/body_models/smpl --pw3d data/3DPW \
      --pred data/hmr2_3dpw_test.npz --field checkpoints/field.pt --mode all
"""
import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloaf.body import SMPLBody  # noqa: E402
from cloaf.collision import collision_rate  # noqa: E402
from cloaf.integrate import correct_sequence  # noqa: E402
from cloaf.metrics import joints14, mpjpe, pa_mpjpe, accel_error  # noqa: E402
from cloaf.model import MotionField  # noqa: E402


def load_3dpw_test(root):
    for f in sorted(glob.glob(os.path.join(root, "sequenceFiles/test/*.pkl"))):
        with open(f, "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        for p in range(len(d["poses"])):
            yield (os.path.basename(f)[:-4], p, d["poses"][p], d["betas"][p][:10],
                   d["genders"][p], d["campose_valid"][p].astype(bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl", required=True)
    ap.add_argument("--pw3d", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--jreg", default="data/J_regressor_h36m.npy")
    ap.add_argument("--field", default=None)
    ap.add_argument("--mode", default="all", choices=["all", "colliding_only"])
    ap.add_argument("--no_cloaf", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = a.device
    body = SMPLBody(a.smpl, device=dev)
    gt_models = {g: SMPLBody(a.smpl, gender={"m": "male", "f": "female"}[g], device=dev)
                 for g in ["m", "f"]}
    Jreg = torch.as_tensor(np.load(a.jreg), device=dev)
    pred = np.load(a.pred, allow_pickle=True)
    field = None
    if not a.no_cloaf:
        field = MotionField().to(dev).eval()
        field.load_state_dict(torch.load(a.field, map_location=dev))

    E = dict(mpjpe=[], pa=[], accel=[], counts=[])
    for seq, p, gt_pose, gt_betas, gender, valid in load_3dpw_test(a.pw3d):
        P = pred[f"{seq}/{p}"].item()
        bp = torch.as_tensor(P["body_pose"], device=dev, dtype=torch.float64)
        go = torch.as_tensor(P["global_orient"], device=dev, dtype=torch.float64)
        be = torch.as_tensor(P["betas"], device=dev, dtype=torch.float64)
        if not a.no_cloaf:
            bp, _ = correct_sequence(body, field, bp, be, go, mode=a.mode)
        with torch.no_grad():
            pv = body.vertices(bp, be, go)
            g = gt_models[gender]
            gp = torch.as_tensor(gt_pose, device=dev, dtype=torch.float64)
            gv = g.vertices(gp[:, 3:], torch.as_tensor(gt_betas, device=dev).expand(len(gp), -1),
                            gp[:, :3])
        v = torch.as_tensor(valid, device=dev)
        pj, gj = joints14(pv, Jreg)[v], joints14(gv, Jreg)[v]
        E["mpjpe"].append(mpjpe(pj, gj) * len(pj))
        E["pa"].append(pa_mpjpe(pj, gj) * len(pj))
        E["accel"].append(accel_error(pj, gj) * (len(pj) - 2))   # TODO: split at gaps in the valid mask
        E["counts"].append(body.count_collisions(pv[v]).cpu())

    n = sum(len(c) for c in E["counts"])
    counts = torch.cat(E["counts"])
    print(f"MPJPE   {sum(E['mpjpe']) / n:6.1f}")
    print(f"P-MPJPE {sum(E['pa']) / n:6.1f}")
    print(f"Accel   {sum(E['accel']) / (n - 2 * len(E['counts'])):6.1f}")
    print(f"Col@0   {collision_rate(counts, 0):6.1f}%   Col@100 {collision_rate(counts, 100):6.1f}%")


if __name__ == "__main__":
    main()
