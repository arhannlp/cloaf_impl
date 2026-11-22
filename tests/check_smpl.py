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
from cloaf.collision import count_collisions  # noqa: E402


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

    c = count_collisions(body.vertices(torch.zeros(1, 69, device=body.device)), body.faces)[0].item()
    print(f"[{'PASS' if c == 0 else 'NOTE'}] T-pose colliding vertices: {c} (expect 0)")
    bent = torch.zeros(1, 69, device=body.device, dtype=body.dtype)
    bent[0, 3 * (18 - 1) + 1] = -2.6        # left elbow fully bent
    bent[0, 3 * (19 - 1) + 1] = 2.6         # right elbow fully bent
    c2 = count_collisions(body.vertices(bent), body.faces)[0].item()
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
