"""
Convert AMASS (SMPL+H npz files) into one training file: body_pose (N,69), seq_id (N,).

  python scripts/prepare_amass.py --amass_root data/amass --out data/amass_smpl_30fps.npz

--amass_root is the folder where you extracted the AMASS sub-datasets
(e.g. data/amass/CMU/..., data/amass/KIT/...). For a quick first run, one
sub-dataset such as CMU is enough. Use --max_files to make it even smaller.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloaf.train import prepare_amass  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amass_root", required=True)
    ap.add_argument("--out", default="data/amass_smpl_30fps.npz")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max_files", type=int, default=None)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = prepare_amass(a.amass_root, a.out, a.fps, a.max_files)
    print(f"wrote {a.out}: {n} frames")


if __name__ == "__main__":
    main()
