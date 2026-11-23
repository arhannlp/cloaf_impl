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

Without a trained network, use a training-free field:
  ... --field point_target                 (Supp. C baseline, no non-intersection property)
  ... --field smooth_target --sigma 0.03   (smooth in space -> touching points move together)

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
from cloaf.integrate import PointTargetField, SmoothTargetField, correct_sequence  # noqa: E402
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
                c = body.count_collisions(body.vertices(p, None, go[peak:peak + 1]))[0].item()
                if best is None or c > best[0]:
                    best = (c, idx, sign)
    c, idx, sign = best
    out = poses.clone()
    out[:, idx] += torch.as_tensor(sign * bump(len(poses), start, length, amp), dtype=out.dtype,
                                   device=out.device)
    print(f"corruption: body_pose[{idx}] {'+' if sign > 0 else '-'}{amp} rad, "
          f"{c} colliding vertices at the peak frame")
    return out


def vertex_dist_mm(body, a, b, go):
    with torch.no_grad():
        return 1000 * (body.vertices(a, None, go) - body.vertices(b, None, go)).norm(dim=-1).mean(-1)


def jerk_mm(body, poses, go):
    """(mean, max) over frames of the mean per-vertex third difference, in mm."""
    with torch.no_grad():
        v = body.vertices(poses, None, go)
    if len(v) < 4:
        return float("nan"), float("nan")
    j = 1000 * (v[3:] - 3 * v[2:-1] + 3 * v[1:-2] - v[:-3]).norm(dim=-1).mean(-1)
    return j.mean().item(), j.max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--field", required=True,
                    help="trained checkpoint (e.g. checkpoints/field.pt), or a training-free "
                         "field: 'point_target' (Supp. C baseline) or 'smooth_target' (see --sigma)")
    ap.add_argument("--sigma", type=float, default=0.03, help="kernel width (m) for smooth_target")
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
    ap.add_argument("--collision_min", type=int, default=50,
                    help="a frame collides if MORE than this many vertices collide (0 = paper's @0); "
                         "50 tolerates the light hand-body contact of clean mocap (7.5 %% of clean "
                         "ACCAD frames exceed 50, 30 %% exceed 0)")
    ap.add_argument("--out", default="results/sequence.npz")
    a = ap.parse_args()

    body = SMPLBody(a.smpl, device=a.device, collision_min=a.collision_min)
    bp, go = load_sequence(a.seq, a.fps, a.start, a.frames)
    clean = torch.as_tensor(bp, device=body.device, dtype=body.dtype)
    go = torch.as_tensor(go, device=body.device, dtype=body.dtype)
    inp = corrupt(body, clean, go, a.corrupt_start, a.corrupt_len, a.corrupt_amp) if a.corrupt else clean

    if a.field == "point_target":
        field = PointTargetField()
    elif a.field == "smooth_target":
        field = SmoothTargetField(a.sigma)
    else:
        field = MotionField().to(body.device).eval()
        field.load_state_dict(torch.load(a.field, map_location=body.device))
    solve_kw = dict(method="rk4", options=dict(step_size=0.1)) if a.solver == "rk4" else {}

    t = time.time()
    out, before = correct_sequence(body, field, inp, None, go, mode=a.mode, S=a.S, progress=True,
                                   **solve_kw)
    secs = time.time() - t
    with torch.no_grad():
        after = body.count_collisions(body.vertices(out, None, go))

    n = len(inp)
    C = a.collision_min
    col_frames = before > C
    shift = vertex_dist_mm(body, out, inp, go)
    print(f"\n{n} frames, {secs:.0f} s ({secs / n:.2f} s/frame), mode={a.mode}, solver={a.solver}")
    print(f"frames with collisions : before {int(col_frames.sum())}/{n}, after {int((after > C).sum())}/{n}"
          f"  (more than {C} colliding vertices)")
    print(f"colliding vertices     : before {int(before.sum())}, after {int(after.sum())} (sum over frames)")
    print(f"vertex shift vs input  : colliding frames {shift[col_frames].mean():.1f} mm, "
          f"clean frames {shift[~col_frames].mean():.1f} mm")
    d_in = d_out = float("nan")
    if a.corrupt:
        d_in = vertex_dist_mm(body, inp, clean, go).mean().item()
        d_out = vertex_dist_mm(body, out, clean, go).mean().item()
        print(f"distance to clean mocap: input {d_in:.1f} mm, CLOAF {d_out:.1f} mm")
    # smoothness: third time-difference of the vertices (sudden jumps show up as peaks)
    jerk = {k: jerk_mm(body, p, go) for k, p in [("clean", clean), ("input", inp), ("cloaf", out)]}
    print("jerk mean / max (mm)   : " + ", ".join(f"{k} {m:.2f} / {x:.2f}" for k, (m, x) in jerk.items()))

    # one row per run -> results/summary_runs.csv, easy to paste into the report
    summary = Path(a.out).parent / "summary_runs.csv"
    new = not summary.exists()
    with open(summary, "a") as fh:
        if new:
            fh.write("sequence,mode,solver,collision_min,frames,col_frames_before,col_frames_after,col_verts_before,"
                     "col_verts_after,shift_colliding_mm,shift_clean_mm,dist_clean_input_mm,"
                     "dist_clean_cloaf_mm,jerk_max_input,jerk_max_cloaf,sec_per_frame\n")
        fh.write(f"{Path(a.seq).stem}[{Path(a.field).stem}],{a.mode},{a.solver},{C},{n},{int(col_frames.sum())},"
                 f"{int((after > C).sum())},{int(before.sum())},{int(after.sum())},"
                 f"{shift[col_frames].mean():.1f},{shift[~col_frames].mean():.1f},{d_in:.1f},{d_out:.1f},"
                 f"{jerk['input'][1]:.2f},{jerk['cloaf'][1]:.2f},{secs / n:.2f}\n")
    print(f"appended a row to {summary}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, clean=clean.cpu().numpy(), input=inp.cpu().numpy(), corrected=out.cpu().numpy(),
             global_orient=go.cpu().numpy(), before=before.cpu().numpy(), after=after.cpu().numpy())
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
