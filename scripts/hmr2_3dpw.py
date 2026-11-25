"""
Run HMR2.0 (4D-Humans) on the 3DPW test set and save per-person SMPL predictions in frame order,
as needed for the paper's Tables 1, 2 and 4 (CLOAF is applied to successive frames of a person).

For every test sequence, every person and every valid frame (campose_valid), the person box is
taken from the 3DPW 2D keypoints (poses2d; fallback: projected 3D joints), the image is cropped
as in the HMR2.0 demo (ViTDetDataset), and HMR2.0 predicts SMPL parameters. Predictions are saved
as axis-angle:

  out.npz['<sequence>/<person>'] = dict(frames (N,), body_pose (N,69), global_orient (N,3), betas (N,10))

The file is written after every sequence and finished sequences are skipped on a re-run, so a
Colab disconnect loses at most one sequence.

Needs 4D-Humans importable (PYTHONPATH) and SMPL_NEUTRAL.pkl in ~/.cache/4DHumans/data/smpl/.

  PYTHONPATH=/content/4D-Humans python scripts/hmr2_3dpw.py --pw3d data/3DPW --out results/hmr2_3dpw_test.npz
"""
import argparse
import glob
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")   # pyrender is imported by hmr2, no display

import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloaf.rotations import rotmat_to_aa  # noqa: E402


def box_from_2d(kp, conf=0.3, pad=1.2):
    """kp (3,18) x, y, confidence -> square box [x0, y0, x1, y1] or None."""
    ok = kp[2] > conf
    if ok.sum() < 4:
        return None
    x, y = kp[0, ok], kp[1, ok]
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    s = max(x.max() - x.min(), y.max() - y.min()) * pad
    return np.array([cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2], dtype=np.float32)


def box_from_3d(joints_world, cam_pose, K, pad=1.2):
    """Project the 24 GT joints (72,) with the camera -> square box."""
    J = joints_world.reshape(24, 3)
    Jc = J @ cam_pose[:3, :3].T + cam_pose[:3, 3]
    if (Jc[:, 2] <= 0).any():
        return None
    uv = Jc @ K.T
    uv = uv[:, :2] / uv[:, 2:]
    kp = np.concatenate([uv.T, np.ones((1, 24))], 0)
    return box_from_2d(kp, conf=0.5, pad=pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pw3d", required=True, help="folder with sequenceFiles/ and imageFiles/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seqs", nargs="*", default=None, help="only these sequences (names)")
    ap.add_argument("--max_frames", type=int, default=None, help="per person, for quick tests")
    a = ap.parse_args()

    import cv2
    import hmr2.datasets.vitdet_dataset as vitdet
    from hmr2.configs import CACHE_DIR_4DHUMANS
    from hmr2.models import DEFAULT_CHECKPOINT, download_models, load_hmr2
    vitdet.print = lambda *args, **kw: None          # the dataset prints one line per crop

    download_models(CACHE_DIR_4DHUMANS)
    # PyTorch >= 2.6 loads checkpoints with weights_only=True, which rejects the omegaconf config
    # stored in the official HMR2.0 checkpoint. The file comes from the 4D-Humans authors, so it is
    # loaded the way it was saved (weights_only=False).
    _torch_load = torch.load

    def _load_full(*args, **kw):
        kw["weights_only"] = False
        return _torch_load(*args, **kw)

    torch.load = _load_full
    try:
        model, cfg = load_hmr2(DEFAULT_CHECKPOINT)
    finally:
        torch.load = _torch_load
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev).eval()

    results = {}
    if os.path.exists(a.out):
        results = {k: v.item() for k, v in np.load(a.out, allow_pickle=True).items()}
        old = [k for k, v in results.items() if v.get("image_index") != "frame"]
        for k in old:              # written by the first version, which read the wrong images
            del results[k]
        print(f"resuming: {len(results)} tracks already in {a.out}"
              + (f" ({len(old)} outdated tracks dropped)" if old else ""))

    def save():
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.out, **{k: np.array(v, dtype=object) for k, v in results.items()})

    files = sorted(glob.glob(os.path.join(a.pw3d, "sequenceFiles", a.split, "*.pkl")))
    if not files:
        raise SystemExit(f"no .pkl in {a.pw3d}/sequenceFiles/{a.split}")
    t0 = time.time()
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        seq = d["sequence"]
        if a.seqs and seq not in a.seqs:
            continue
        for p in range(len(d["poses"])):
            key = f"{seq}/{p}"
            if key in results:
                continue
            frames = np.nonzero(np.asarray(d["campose_valid"][p]).astype(bool))[0]
            if a.max_frames:
                frames = frames[:a.max_frames]
            used, preds, items, missing = [], [], [], 0

            def run(items):
                batch = torch.utils.data.default_collate(items)
                batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                with torch.no_grad():
                    sp = model(batch)["pred_smpl_params"]
                return (rotmat_to_aa(sp["body_pose"].reshape(-1, 23, 3, 3)).reshape(-1, 69).cpu(),
                        rotmat_to_aa(sp["global_orient"].reshape(-1, 3, 3)).cpu(),
                        sp["betas"].reshape(-1, 10).double().cpu())

            for t in frames:
                # poses[p][t] belongs to image_{t:05d}.jpg (as in SPIN / VIBE preprocessing);
                # img_frame_ids is not an index into imageFiles
                path = os.path.join(a.pw3d, "imageFiles", seq, f"image_{int(t):05d}.jpg")
                img = cv2.imread(path) if os.path.exists(path) else None
                if img is None:
                    missing += 1
                    continue
                box = box_from_2d(np.asarray(d["poses2d"][p][t]))
                if box is None:
                    box = box_from_3d(np.asarray(d["jointPositions"][p][t]), np.asarray(d["cam_poses"][t]),
                                      np.asarray(d["cam_intrinsics"]))
                if box is None:
                    continue
                items.append(vitdet.ViTDetDataset(cfg, img, box[None])[0])
                used.append(int(t))
                if len(items) == a.batch:
                    preds.append(run(items))
                    items = []
            if items:
                preds.append(run(items))
            if not preds:
                print(f"{key}: no frames")
                continue
            bp, go, be = (torch.cat([x[i] for x in preds]).numpy() for i in range(3))
            results[key] = dict(frames=np.array(used), body_pose=bp, global_orient=go, betas=be,
                                image_index="frame")
            print(f"{key}: {len(used)}/{len(frames)} frames"
                  + (f", {missing} images missing" if missing else "")
                  + f"  ({time.time() - t0:.0f} s)", flush=True)
        save()
    print(f"saved {a.out}: {len(results)} tracks, {sum(len(v['frames']) for v in results.values())} frames")


if __name__ == "__main__":
    main()
