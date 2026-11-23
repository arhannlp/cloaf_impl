"""
The working CLOAF pipeline with the settings chosen in our experiments (docs/report_template.md).

    from cloaf.body import SMPLBody
    from cloaf.pipeline import correct
    body = SMPLBody("data/body_models/smpl", device="cuda", collision_min=50)
    result = correct(body, body_pose, global_orient=go, betas=betas)
    result["body_pose"]            # corrected (N, 69)

Ingredients (paper section / our extension):
  - flow towards each estimated pose, integrated in the SMPL parameters (Eq. 6), RK4, 10 steps
  - field: "smooth_target" (training-free, a smooth function of position) or a trained
    MotionField checkpoint (Eq. 8)
  - contact-constrained projection instead of the plain J^+ (extension 2): points of
    non-adjacent body parts closer than 5 cm may not approach each other
  - successive frames (Sec. 4.2), only colliding frames corrected, with a return phase that
    fades the correction out (extension 3) and temporal smoothing (extension 4)
  - start pose: previous output, last collision-free output, jitter (Sec. 4.5), rest pose
"""
from __future__ import annotations

import time

import torch

from .integrate import PointTargetField, SmoothTargetField, correct_sequence
from .model import MotionField

DEFAULTS = dict(
    field="smooth_target",   # or "point_target", or a path to a trained MotionField checkpoint
    sigma=0.03,              # kernel width of smooth_target (m)
    mode="colliding_only",   # or "all" (paper: every frame)
    contact_dist=0.05,       # contact constraint distance (m); 0 = paper's plain J^+
    return_eps=0.005,        # return phase: stop when the faded correction is below 5 mm
    return_decay=0.7,        # return phase: correction kept per frame
    smooth_sigma=1.5,        # temporal smoothing of the corrected frames (frames)
    S=1000,                  # surface points
    steps=10,                # RK4 steps per frame
)


def make_field(field, sigma=0.03, device="cpu"):
    if field == "smooth_target":
        return SmoothTargetField(sigma)
    if field == "point_target":
        return PointTargetField()
    net = MotionField().to(device).eval()
    net.load_state_dict(torch.load(field, map_location=device))
    return net


def jerk_mm(body, poses, betas=None, global_orient=None):
    """Per-frame jerk (mean per-vertex third difference, mm); (N-3,)."""
    with torch.no_grad():
        v = body.vertices(poses, betas, global_orient)
    if len(v) < 4:
        return torch.zeros(0)
    return 1000 * (v[3:] - 3 * v[2:-1] + 3 * v[1:-2] - v[:-3]).norm(dim=-1).mean(-1)


@torch.no_grad()
def correct(body, body_pose, global_orient=None, betas=None, progress=True, **options):
    """Correct the self-intersections of a pose sequence.

    body_pose (N,69), global_orient (N,3) or None, betas (N,10) or None.
    options override DEFAULTS. Returns a dict with the corrected body_pose, the colliding
    vertices per frame before/after, and summary numbers."""
    o = {**DEFAULTS, **options}
    dev, dt = body.device, body.dtype
    as_t = lambda x: None if x is None else torch.as_tensor(x, device=dev, dtype=dt)
    poses, go, be = as_t(body_pose), as_t(global_orient), as_t(betas)
    field = make_field(o["field"], o["sigma"], dev)

    t = time.time()
    out, before = correct_sequence(
        body, field, poses, be, go, mode=o["mode"], S=o["S"], progress=progress,
        return_eps=o["return_eps"] or None, return_decay=o["return_decay"],
        smooth_sigma=o["smooth_sigma"] or None,
        method="rk4", options=dict(step_size=1.0 / o["steps"]), contact_dist=o["contact_dist"])
    seconds = time.time() - t
    after = body.count_collisions(body.vertices(out, be, go))
    shift = 1000 * (body.vertices(out, be, go) - body.vertices(poses, be, go)).norm(dim=-1).mean(-1)
    j_in, j_out = jerk_mm(body, poses, be, go), jerk_mm(body, out, be, go)
    C = body.collision_min
    return dict(
        body_pose=out, before=before, after=after, shift_mm=shift, seconds=seconds, options=o,
        summary=dict(
            frames=len(poses),
            colliding_frames_before=int((before > C).sum()),
            colliding_frames_after=int((after > C).sum()),
            colliding_vertices_before=int(before.sum()),
            colliding_vertices_after=int(after.sum()),
            mean_shift_mm=float(shift.mean()),
            max_jerk_before_mm=float(j_in.max()) if len(j_in) else 0.0,
            max_jerk_after_mm=float(j_out.max()) if len(j_out) else 0.0,
            seconds_per_frame=seconds / max(1, len(poses)),
        ))
