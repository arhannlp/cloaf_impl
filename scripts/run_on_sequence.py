"""
Run CLOAF on ONE motion sequence and report before/after numbers.
This is the first task of the project: "run CLOAF on a small selection of SMPL
motion sequences and examine the results as animations".

Input: one AMASS sequence file (.npz with 'poses'), or a .npz you saved yourself
with 'body_pose' (N,69) and optionally 'global_orient' (N,3).

Clean mocap rarely has self-intersections, so --corrupt injects them: it
smoothly rotates one shoulder into the torso over a window of frames. Because
the clean motion is known, you can also measure how close CLOAF gets to it.

  python scripts/run_on_sequence.py --smpl data/body_models/smpl \
      --seq data/amass/CMU/01/01_01_poses.npz --field checkpoints/field.pt \
      --frames 90 --corrupt --mode all

Then watch the result:
  python scripts/view_results.py --smpl data/body_models/smpl --result results/sequence.npz
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloaf.body import SMPLBody, L_SHOULDER, R_SHOULDER  # noqa: E402
from cloaf.collision import count_collisions  # noqa: E402
from cloaf.integrate import correct_sequence  # noqa: E402
from cloaf.model import MotionField  # noqa: E402


def load_sequence(path, fps, start, frames):
    d = np.load(path)
    if "poses" in d:  # AMASS SMPL+H
        src = float(d["mocap_framerate"] if "mocap_framerate" in d else d["mocap_frame_rate"])
        step = max(1, int(round(src / fps)))
        p = d["poses"][::step]
        body = np.concatenate([p[:, 3:66], np.zeros((len(p), 6))], 1)
        go = p[:, :3]
    else:
        body = d["body_pose"]
        go = d["global_orient"] if "global_orient" in d else np.zeros((len(body), 3))
    sl = slice(start, start + frames)
    return body[sl], go[sl]


def bump(n, start, length, amp):
    """Smooth 0 -> amp -> 0 profile over [start, start+length)."""
    w = np.zeros(n)
    k = np.arange(length)
    w[start:start + length] = amp * np.sin(np.pi * (k + 0.5) / length) ** 2
    return w


@torch.no_grad()
def corrupt(body, poses, go, start, length, amp):
    """Try both shoulders, 3 axes, 2 signs; keep the one that creates the most
    colliding vertices at the peak frame."""
    peak = start + length // 2
    best = None
    for joint in (L_SHOULDER, R_SHOULDER):
        for axis in range(3):
            for sign in (1, -1):
                idx = 3 * (joint - 1) + axis
                p = poses[peak:peak + 1].clone()
                p[0, idx] += sign * amp
                c = count_collisions(body.vertices(p, None, go[peak:peak + 1]), body.faces)[0].item()
                if best is None or c > best[0]:
                    best = (c, idx, sign)
    c, idx, sign = best
    out = poses.clone()
    out[:, idx] += torch.as_tensor(sign * bump(len(poses), start, length, amp), dtype=out.dtype)
    print(f"corruption: body_pose[{idx}] {'+' if sign > 0 else '-'}{amp} rad, "
          f"{c} colliding vertices at the peak frame")
    return out


def vertex_dist_mm(body, a, b, go):
    with torch.no_grad():
        return 1000 * (body.vertices(a, None, go) - body.vertices(b, None, go)).norm(dim=-1).mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--field", required=True, help="trained checkpoint, e.g. checkpoints/field.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--corrupt", action="store_true")
    ap.add_argument("--corrupt_start", type=int, default=30)
    ap.add_argument("--corrupt_len", type=int, default=30)
    ap.add_argument("--corrupt_amp", type=float, default=1.2)
    ap.add_argument("--mode", default="all", choices=["all", "colliding_only"])
    ap.add_argument("--solver", default="rk4", choices=["rk4", "dopri5"],
                    help="rk4 = fixed 10 steps (fast); dopri5 = paper's adaptive solver (slow)")
    ap.add_argument("--S", type=int, default=1000)
    ap.add_argument("--out", default="results/sequence.npz")
    a = ap.parse_args()

    body = SMPLBody(a.smpl, device=a.device)
    bp, go = load_sequence(a.seq, a.fps, a.start, a.frames)
    clean = torch.as_tensor(bp, device=body.device, dtype=body.dtype)
    go = torch.as_tensor(go, device=body.device, dtype=body.dtype)
    inp = corrupt(body, clean, go, a.corrupt_start, a.corrupt_len, a.corrupt_amp) if a.corrupt else clean

    field = MotionField().to(body.device).eval()
    field.load_state_dict(torch.load(a.field, map_location=body.device))
    solve_kw = dict(method="rk4", options=dict(step_size=0.1)) if a.solver == "rk4" else {}

    t = time.time()
    out, before = correct_sequence(body, field, inp, None, go, mode=a.mode, S=a.S, **solve_kw)
    secs = time.time() - t
    with torch.no_grad():
        after = count_collisions(body.vertices(out, None, go), body.faces)

    n = len(inp)
    col_frames = before > 0
    shift = vertex_dist_mm(body, out, inp, go)
    print(f"\n{n} frames, {secs:.0f} s ({secs / n:.2f} s/frame), mode={a.mode}, solver={a.solver}")
    print(f"frames with collisions : before {int(col_frames.sum())}/{n}, after {int((after > 0).sum())}/{n}")
    print(f"colliding vertices     : before {int(before.sum())}, after {int(after.sum())} (sum over frames)")
    print(f"vertex shift vs input  : colliding frames {shift[col_frames].mean():.1f} mm, "
          f"clean frames {shift[~col_frames].mean():.1f} mm")
    if a.corrupt:
        print(f"distance to clean mocap: input {vertex_dist_mm(body, inp, clean, go).mean():.1f} mm, "
              f"CLOAF {vertex_dist_mm(body, out, clean, go).mean():.1f} mm")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, clean=clean.cpu().numpy(), input=inp.cpu().numpy(), corrected=out.cpu().numpy(),
             global_orient=go.cpu().numpy(), before=before.cpu().numpy(), after=after.cpu().numpy())
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
