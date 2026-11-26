#!/bin/bash
# Train the motion field on a Google Cloud GPU VM, with data and checkpoints in a Cloud Storage bucket.
#
#   bash scripts/gcp_train.sh gs://YOUR_BUCKET 600      # train for 600 minutes in total
#
# The bucket must contain (copied from Drive with the notebook cell "G1"):
#   SMPL_python_v.1.1.0.zip, amass/*.tar.bz2, checkpoints/field.pt* (optional: continue training)
# Every hour the checkpoint is copied back to gs://YOUR_BUCKET/checkpoints/, so stopping the VM
# (or a crash) loses at most one hour. Running the script again continues from the bucket.
# Run it inside tmux so that closing the SSH window does not stop it:
#   tmux new -s train      (detach: Ctrl-b d, re-attach: tmux attach -t train)
set -euo pipefail
B=${1:?usage: gcp_train.sh gs://BUCKET [total_minutes] [test_set]}
TOTAL=${2:-600}
TEST_SET=${3:-SFU}          # held-out AMASS subset, not used for training
cd "$(dirname "$0")/.."

pip install -q torchdiffeq smplx trimesh
mkdir -p data/body_models data/amass ckpt

if [ ! -f data/body_models/smpl/SMPL_NEUTRAL.pkl ]; then
    gsutil cp "$B/SMPL_python_v.1.1.0.zip" data/
    python scripts/convert_smpl_pkl.py --src data/SMPL_python_v.1.1.0.zip --out data/body_models/smpl
fi
if [ ! -f data/amass_smpl_30fps.npz ]; then
    gsutil -m cp "$B/amass/*.tar.bz2" data/
    for f in data/*.tar.bz2; do echo "extracting $f"; tar -xjf "$f" -C data/amass; done
    ROOTS=$(find data/amass -mindepth 1 -maxdepth 1 -type d ! -name "$TEST_SET" | sort | tr '\n' ' ')
    echo "training sets: $ROOTS (held out: $TEST_SET)"
    python scripts/prepare_amass.py --amass_root $ROOTS --out data/amass_smpl_30fps.npz
fi
gsutil -m cp "$B/checkpoints/field.pt*" ckpt/ 2>/dev/null || echo "no checkpoint in the bucket: training from scratch"

nvidia-smi --query-gpu=name,memory.total --format=csv
done_min=0
while [ "$done_min" -lt "$TOTAL" ]; do
    python -u -m cloaf.train --smpl data/body_models/smpl --amass data/amass_smpl_30fps.npz \
        --batch 16 --minutes 60 --resume --out ckpt/field.pt
    gsutil -m cp ckpt/field.pt ckpt/field.pt.state ckpt/field.pt.log.csv "$B/checkpoints/"
    done_min=$((done_min + 60))
    echo "=== $done_min / $TOTAL minutes, checkpoint copied to $B/checkpoints/ ==="
done
python -u scripts/loss_reference.py --smpl data/body_models/smpl --amass data/amass_smpl_30fps.npz \
    --field ckpt/field.pt 2>&1 | grep -v WARNING | tee ckpt/loss_reference.txt
gsutil cp ckpt/loss_reference.txt "$B/checkpoints/"
echo "finished. Stop the VM now (Compute Engine -> VM instances -> Stop) to stop paying."
