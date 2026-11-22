"""
Quick check that the real SMPL files load and behave (about 1 minute on a CPU).

  python tests/check_smpl.py --smpl data/body_models/smpl
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloaf.body import SMPLBody  # noqa: E402
from cloaf.collision import winding_numbers  # noqa: E402

JOINTS = ['pelvis', 'l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee', 'spine2', 'l_ankle', 'r_ankle',
          'spine3', 'l_foot', 'r_foot', 'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder',
          'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl", default="data/body_models/smpl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    for gender in ("neutral", "male", "female"):
        try:
            body = SMPLBody(a.smpl, gender=gender, device=a.device)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {gender}: {type(e).__name__}: {e}")
            if "chumpy" in str(e):
                print("       -> run scripts/convert_smpl_pkl.py first")
            continue
        v = body.vertices(torch.zeros(1, 69, device=body.device))
        ok = v.shape[1] == 6890 and body.faces.shape[0] == 13776
        print(f"[{'PASS' if ok else 'FAIL'}] {gender}: {v.shape[1]} vertices, {body.faces.shape[0]} faces")

    body = SMPLBody(a.smpl, device=a.device)
    try:
        import trimesh
        m = trimesh.Trimesh(body.vertices(torch.zeros(1, 69, device=body.device))[0].cpu().numpy(),
                            body.faces.cpu().numpy(), process=False)
        print(f"[{'PASS' if m.is_watertight else 'NOTE'}] mesh watertight: {m.is_watertight} "
              "(the winding-number collision test assumes a closed mesh)")
    except ImportError:
        print("[SKIP] trimesh not installed, watertight check skipped")

    # Collision test in the rest (T) pose: should be 0 once the rest-pose vertices are ignored
    v0 = body.vertices(torch.zeros(1, 69, device=body.device))[0]
    w = winding_numbers(v0, body.faces)
    rest = body.rest_inside
    print(f"[INFO] collision test '{body.collision_mode}': {int(rest.sum())} vertices flagged in the "
          f"rest pose (full-mesh winding number max {w.max():.3f}); they are ignored when counting")
    print(f"[INFO] ignored vertices in total (rest pose + crotch/armpit regions): "
          f"{int(body.collision_ignore.sum())}")
    if rest.any():
        J = body.smpl.lbs_weights[rest].argmax(-1)
        names = {}
        for j in J.tolist():
            names[JOINTS[j]] = names.get(JOINTS[j], 0) + 1
        print("       by body part:", dict(sorted(names.items(), key=lambda x: -x[1])))
        print("       winding numbers:", [round(x, 3) for x in w[rest][:10].tolist()])
    c = body.count_collisions(v0[None])[0].item()
    print(f"[{'PASS' if c == 0 else 'FAIL'}] T-pose colliding vertices after ignoring: {c} (expect 0)")
    bent = torch.zeros(1, 69, device=body.device, dtype=body.dtype)
    bent[0, 3 * (18 - 1) + 1] = -2.6        # left elbow fully bent
    bent[0, 3 * (19 - 1) + 1] = 2.6         # right elbow fully bent
    c2 = body.count_collisions(body.vertices(bent))[0].item()
    print(f"[{'PASS' if c2 > 0 else 'NOTE'}] fully bent elbows, colliding vertices: {c2} "
          "(expect > 0: forearm inside upper arm)")

    sample = body.sample_surface(1000)
    th = torch.zeros(1, 69, device=body.device, dtype=body.dtype)
    t = time.perf_counter()
    J = body.jacobian(th, sample)
    print(f"[PASS] Jacobian {tuple(J.shape)} in {1000 * (time.perf_counter() - t):.0f} ms on {a.device}")
    print("excluded (crotch/armpit) vertices:", int(body.excluded_verts.sum()))


if __name__ == "__main__":
    main()
