"""
3DPW-test evaluation as in the paper (Table 1, 2, 4): MPJPE, P-MPJPE, Accel.Err, Col.Rate@0/@100,
for the HMR2.0 predictions (scripts/hmr2_3dpw.py) and, optionally, after correcting them with CLOAF.

  # HMR2.0 row (Table 1 / 2)
  python scripts/eval_3dpw.py --smpl data/body_models/smpl --pw3d data/3DPW --pred results/hmr2_3dpw_test.npz

  # CLOAF as in the paper: successive frames, every frame, plain J^+ projection
  python scripts/eval_3dpw.py ... --cloaf paper --field checkpoints/field.pt

Paper values (3DPW-test):          MPJPE  P-MPJPE  Accel  Col@0
  HMR2.0                            82.0    52.7    16.1   39.2 %   (Col@100 21.6 %, Table 1)
  CLOAF (successive frames)         82.3    53.2     9.4    0.0 %
  CLOAF jitter / keyposes          81.6 / 82.9  52.6 / 53.6  15.9 / 16.4   (Table 4)

Protocol (as in SPIN / VIBE / HMR2.0):
  - ground truth: gendered SMPL with the 3DPW poses, global orientation rotated into the camera
    frame (cam_poses), only frames with campose_valid
  - prediction: neutral SMPL with the predicted parameters
  - joints: 14 LSP joints from J_regressor_h36m.npy if given (standard), else from 4D-Humans'
    SMPL_to_J19.pkl, else the 24 SMPL joints (then the numbers are not comparable to the paper);
    pelvis-aligned (mid-hip); P-MPJPE after Procrustes alignment; mm
  - Accel: difference of joint accelerations (mm/frame^2) within runs of consecutive frames
  - Col.Rate@C: % of frames with more than C colliding vertices (our collision test, see
    cloaf/collision.py; the paper does not describe its own)
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloaf.body import SMPLBody  # noqa: E402
from cloaf.collision import collision_rate  # noqa: E402
from cloaf.metrics import procrustes  # noqa: E402
from cloaf.rotations import aa_to_rotmat, rotmat_to_aa  # noqa: E402

H36M_TO_J14 = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 10]
PAPER = {"HMR2.0": (82.0, 52.7, 16.1, 39.2, 21.6), "CLOAF": (82.3, 53.2, 9.4, 0.0, None)}


def make_joints(a, device):
    """Joint regressor -> (function verts (N,V,3) -> joints (N,J,3), hip indices, name)."""
    if a.jreg and os.path.exists(a.jreg):
        J = torch.as_tensor(np.load(a.jreg), device=device, dtype=torch.float64)
        return (lambda v: torch.einsum("jv,nvc->njc", J.to(v), v)[:, H36M_TO_J14]), (2, 3), "H36M regressor, 14 LSP joints"
    j19 = a.j19 or os.path.expanduser("~/.cache/4DHumans/data/SMPL_to_J19.pkl")
    if os.path.exists(j19):
        with open(j19, "rb") as fh:
            J = pickle.load(fh, encoding="latin1")
        J = torch.as_tensor(np.asarray(J.todense() if hasattr(J, "todense") else J)[:14],
                            device=device, dtype=torch.float64)
        return (lambda v: torch.einsum("jv,nvc->njc", J.to(v), v)), (2, 3), "4D-Humans SMPL_to_J19, 14 LSP joints"
    return None, (1, 2), "SMPL 24 joints (NOT comparable to the paper)"


def per_frame_errors(pj, gj, hips):
    """MPJPE, P-MPJPE per frame (mm)."""
    center = lambda j: j - 0.5 * (j[:, hips[0]:hips[0] + 1] + j[:, hips[1]:hips[1] + 1])
    mp = 1000 * (center(pj) - center(gj)).norm(dim=-1).mean(-1)
    pa = 1000 * (procrustes(pj, gj) - gj).norm(dim=-1).mean(-1)
    return mp, pa


def accel_errors(pj, gj, frames, hips):
    """Accel error (mm/frame^2) for every frame that has both neighbours in its run."""
    center = lambda j: j - 0.5 * (j[:, hips[0]:hips[0] + 1] + j[:, hips[1]:hips[1] + 1])
    p, g = center(pj), center(gj)
    out = []
    breaks = np.nonzero(np.diff(frames) != 1)[0] + 1
    for s, e in zip(np.r_[0, breaks], np.r_[breaks, len(frames)]):
        if e - s < 3:
            continue
        ap = p[s:e - 2] - 2 * p[s + 1:e - 1] + p[s + 2:e]
        ag = g[s:e - 2] - 2 * g[s + 1:e - 1] + g[s + 2:e]
        out.append(1000 * (ap - ag).norm(dim=-1).mean(-1))
    return torch.cat(out) if out else torch.zeros(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smpl", required=True)
    ap.add_argument("--pw3d", required=True)
    ap.add_argument("--pred", required=True, help="output of scripts/hmr2_3dpw.py")
    ap.add_argument("--jreg", default="data/J_regressor_h36m.npy")
    ap.add_argument("--j19", default=None, help="4D-Humans SMPL_to_J19.pkl (default: its cache)")
    ap.add_argument("--cloaf", default="none", choices=["none", "paper", "ours"],
                    help="none: HMR2.0 only; paper: successive frames, every frame, plain J^+; "
                         "ours: the settings of cloaf/pipeline.py")
    ap.add_argument("--field", default=None, help="trained checkpoint (paper) or smooth_target")
    ap.add_argument("--steps", type=int, default=10, help="RK4 steps per frame")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--max_frames", type=int, default=None, help="per track, for quick tests")
    ap.add_argument("--out", default=None, help="save corrected poses and per-frame metrics (.npz)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    dev, f64 = a.device, torch.float64
    body = SMPLBody(a.smpl, device=dev, collision_min=0 if a.cloaf == "paper" else 50)
    gt_models = {g: SMPLBody(a.smpl, gender=n, device=dev) for g, n in (("m", "male"), ("f", "female"))}
    joints_fn, hips, jname = make_joints(a, dev)
    if joints_fn is None:
        joints_fn = lambda v: torch.einsum("jv,nvc->njc", body.smpl.J_regressor.to(v), v)[:, :24]
    print(f"joints: {jname}", flush=True)
    pred = {k: v.item() for k, v in np.load(a.pred, allow_pickle=True).items()}

    if a.cloaf != "none":
        from cloaf.pipeline import correct
        if a.cloaf == "paper":
            opts = dict(field=a.field or "smooth_target", mode="all", contact_dist=0.0, return_eps=0,
                        smooth_sigma=0, steps=a.steps)
        else:
            opts = dict(field=a.field or "smooth_target", steps=a.steps)

    M = dict(mpjpe=[], pa=[], accel=[], counts=[])
    saved = {}
    # per-track results are cached next to --out, so an interrupted run continues where it stopped
    cache_path = (a.out[:-4] if a.out and a.out.endswith(".npz") else (a.out or "")) + ".partial.npz"
    cache = {}
    if a.out and os.path.exists(cache_path):
        cache = {k: v.item() for k, v in np.load(cache_path, allow_pickle=True).items()}
        cache = {k: v for k, v in cache.items() if v.get("joints") == jname}   # same joints only
        print(f"resuming: {len(cache)} tracks already evaluated in {cache_path}")
    t0 = time.time()
    for f in sorted(glob.glob(os.path.join(a.pw3d, "sequenceFiles", "test", "*.pkl"))):
        with open(f, "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        seq = d["sequence"]
        if a.seqs and seq not in a.seqs:
            continue
        for p in range(len(d["poses"])):
            key = f"{seq}/{p}"
            if key not in pred:
                print(f"{key}: no prediction, skipped")
                continue
            P = pred[key]
            fr = np.asarray(P["frames"])
            if a.max_frames:
                fr = fr[:a.max_frames]
            if key in cache:
                c = cache[key]
                for k in M:
                    M[k].append(torch.as_tensor(c[k]))
                saved[key] = c["saved"]
                continue
            n = len(fr)
            bp = torch.as_tensor(P["body_pose"][:n], device=dev, dtype=f64)
            go = torch.as_tensor(P["global_orient"][:n], device=dev, dtype=f64)
            be = torch.as_tensor(P["betas"][:n], device=dev, dtype=f64)
            if a.cloaf != "none":
                r = correct(body, bp, go, be, progress=False, **opts)
                bp = r["body_pose"]

            # ground truth in the camera frame
            gp = torch.as_tensor(np.asarray(d["poses"][p])[fr], device=dev, dtype=f64)
            Rc = torch.as_tensor(np.asarray(d["cam_poses"])[fr][:, :3, :3], device=dev, dtype=f64)
            go_gt = rotmat_to_aa(Rc @ aa_to_rotmat(gp[:, :3]))
            gb = torch.as_tensor(np.asarray(d["betas"][p])[:10], device=dev, dtype=f64).expand(n, -1)
            g = gt_models[d["genders"][p][0].lower()]
            with torch.no_grad():
                pv = body.vertices(bp, be, go)
                gv = g.vertices(gp[:, 3:], gb, go_gt)
                pj, gj = joints_fn(pv), joints_fn(gv)
                mp, pa = per_frame_errors(pj, gj, hips)
                acc = accel_errors(pj, gj, fr, hips)
                counts = body.count_collisions(pv)
            M["mpjpe"].append(mp.cpu()); M["pa"].append(pa.cpu())
            M["accel"].append(acc.cpu()); M["counts"].append(counts.cpu())
            if a.out:
                saved[key] = dict(frames=fr, body_pose=bp.cpu().numpy(), global_orient=go.cpu().numpy(),
                                  betas=be.cpu().numpy(), mpjpe=mp.cpu().numpy(), counts=counts.cpu().numpy())
            if a.out:
                cache[key] = dict(mpjpe=mp.cpu().numpy(), pa=pa.cpu().numpy(), accel=acc.cpu().numpy(),
                                  counts=counts.cpu().numpy(), saved=saved[key], joints=jname)
                Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache_path, **{k: np.array(v, dtype=object) for k, v in cache.items()})
            print(f"{key}: {n} frames, MPJPE {mp.mean():.1f}, P-MPJPE {pa.mean():.1f}, "
                  f"Col@0 {collision_rate(counts, 0):.1f} %  ({time.time() - t0:.0f} s)", flush=True)

    cat = {k: torch.cat(v) for k, v in M.items()}
    res = dict(frames=len(cat["mpjpe"]), MPJPE=cat["mpjpe"].mean().item(), P_MPJPE=cat["pa"].mean().item(),
               Accel=cat["accel"].mean().item(), Col0=collision_rate(cat["counts"], 0),
               Col100=collision_rate(cat["counts"], 100), joints=jname, cloaf=a.cloaf)
    name = "HMR2.0" if a.cloaf == "none" else f"CLOAF ({a.cloaf})"
    ref = PAPER["HMR2.0" if a.cloaf == "none" else "CLOAF"]
    print(f"\n{'':20s} {'MPJPE':>7s} {'P-MPJPE':>8s} {'Accel':>6s} {'Col@0':>7s} {'Col@100':>8s}  frames")
    print(f"{name:20s} {res['MPJPE']:7.1f} {res['P_MPJPE']:8.1f} {res['Accel']:6.1f} "
          f"{res['Col0']:6.1f}% {res['Col100']:7.1f}%  {res['frames']}")
    print(f"{'paper':20s} {ref[0]:7.1f} {ref[1]:8.1f} {ref[2]:6.1f} {ref[3]:6.1f}% "
          f"{'' if ref[4] is None else f'{ref[4]:7.1f}%':>8s}")
    print(json.dumps(res))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.out, summary=json.dumps(res), **{k: np.array(v, dtype=object) for k, v in saved.items()})
        print(f"saved {a.out}")


if __name__ == "__main__":
    main()
