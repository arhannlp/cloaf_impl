"""
Correct the self-intersections of an SMPL motion sequence with CLOAF (one command).

  python scripts/cloaf_correct.py --smpl data/body_models/smpl \
      --input my_motion.npz --output my_motion_cloaf.npz

Input (.npz), either
  - an AMASS file (key 'poses', SMPL+H or SMPL-X; resampled to --fps), or
  - your own file with 'body_pose' (N,69) and optionally 'global_orient' (N,3),
    'betas' (N,10) or (10,), 'transl' (N,3).
Output (.npz): 'corrected' / 'input' body poses (N,69), 'global_orient', 'betas', 'transl',
colliding vertices per frame 'before' / 'after'. Watch it with scripts/render_gif.py
(in Colab) or scripts/view_results.py (aitviewer).

Defaults are the settings chosen in our experiments (cloaf/pipeline.py DEFAULTS); every
one of them can be changed on the command line.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloaf.body import SMPLBody  # noqa: E402
from cloaf.pipeline import DEFAULTS, correct  # noqa: E402


def load_motion(path, fps):
    d = np.load(path)
    if "poses" in d:                                   # AMASS (SMPL+H or SMPL-X)
        src = float(d["mocap_framerate"] if "mocap_framerate" in d else d["mocap_frame_rate"])
        step = max(1, int(round(src / fps)))
        p = d["poses"][::step]
        body = np.concatenate([p[:, 3:66], np.zeros((len(p), 6))], 1)
        transl = d["trans"][::step] if "trans" in d else None
        # AMASS shape coefficients belong to SMPL+H/SMPL-X, not SMPL: use the mean shape
        return body, p[:, :3], None, transl
    body = d["body_pose"]
    go = d["global_orient"] if "global_orient" in d else None
    betas = d["betas"] if "betas" in d else None
    if betas is not None:
        betas = np.broadcast_to(betas[..., :10], (len(body), 10)).copy()
    transl = d["transl"] if "transl" in d else None
    return body, go, betas, transl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smpl", required=True, help="folder with SMPL_NEUTRAL.pkl (and MALE/FEMALE)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    ap.add_argument("--fps", type=int, default=30, help="AMASS input is resampled to this rate")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=None, help="only this many frames from --start")
    ap.add_argument("--collision_min", type=int, default=50,
                    help="a frame collides with more than this many colliding vertices")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v, help=f"default {v}")
    a = ap.parse_args()

    body_pose, go, betas, transl = load_motion(a.input, a.fps)
    sl = slice(a.start, None if a.frames is None else a.start + a.frames)
    body_pose = body_pose[sl]
    go = None if go is None else go[sl]
    betas = None if betas is None else betas[sl]
    transl = None if transl is None else transl[sl]
    print(f"{a.input}: {len(body_pose)} frames", flush=True)

    body = SMPLBody(a.smpl, gender=a.gender, device=a.device, collision_min=a.collision_min)
    options = {k: getattr(a, k) for k in DEFAULTS}
    r = correct(body, body_pose, go, betas, **options)

    s = r["summary"]
    print(json.dumps(s, indent=2))
    print(f"colliding frames: {s['colliding_frames_before']} -> {s['colliding_frames_after']} "
          f"of {s['frames']} (more than {a.collision_min} colliding vertices)")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, corrected=r["body_pose"].cpu().numpy(), input=np.asarray(body_pose),
             global_orient=np.zeros((len(body_pose), 3)) if go is None else np.asarray(go),
             betas=np.zeros((len(body_pose), 10)) if betas is None else np.asarray(betas),
             transl=np.zeros((len(body_pose), 3)) if transl is None else np.asarray(transl),
             before=r["before"].cpu().numpy(), after=r["after"].cpu().numpy(),
             summary=json.dumps(s), options=json.dumps(r["options"]))
    print(f"saved {a.output}")


if __name__ == "__main__":
    main()
