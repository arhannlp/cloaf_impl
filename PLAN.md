# Plan: reproduce the experiments of the paper

Supervisor's guidance: focus on reproducing the paper's experiments first; no experiments beyond the paper yet. The extensions already explored (contact constraint, return phase, smoothing; `docs/report_template.md` Sec. 5b–5d) are kept for later.

| Paper experiment | Data | Status |
|---|---|---|
| Fig. 5, Fig. B.7/B.8: Jacobian accuracy vs. points and step size | SMPL | done, matches |
| Fig. 3, Fig. 4: simple fields, moving the arm to a target, box obstacle (Sec. 4.4) | SMPL | code done (`scripts/demo_target_field.py`); figures to make |
| Table 1: Col.Rate of SOTA methods on 3DPW-test | 3DPW + HMR2.0 (+ PARE, TUCH, EFT, SPIN) | HMR2.0 done (below); others later |
| Table 2: CLOAF vs. HMR2.0 (MPJPE, P-MPJPE, Accel, Col.Rate) | 3DPW + HMR2.0 + trained network | needs the trained network |
| Table 4: start pose strategies (successive frames, jitter, keyposes) | 3DPW + HMR2.0, AMASS for keyposes | keyposes to implement |
| Table 3: fine-tuning EFT with CLOAF | 3DPW-train, COCO-EFT | last, optional |

## HMR2.0 on 3DPW-test (35,515 frames, 37 tracks)

|  | MPJPE | P-MPJPE | Accel | Col@0 | Col@100 |
|---|---|---|---|---|---|
| ours | 80.9 | 53.9 | 16.8 | 48.1 % | 23.2 % |
| paper | 82.0 | 52.7 | 16.1 | 39.2 % | 21.6 % |

Accuracy and Col@100 match; our test flags more frames with only a few colliding vertices (Col@0). See `scripts/col_curve.py`.

## Steps

1. **HMR2.0 on 3DPW-test** (Colab T4 is enough): notebook cells R1–R3. Gives the HMR2.0 row of Tables 1 and 2 and calibrates our collision test against the paper's 39.2 % Col@0.
2. **Train the network properly** (Google Cloud student credits, GPU quota requested): several AMASS subsets, validation with `scripts/loss_reference.py`, until the loss approaches the point-target reference.
3. **Table 2**: `scripts/eval_3dpw.py --cloaf paper --field checkpoint.pt`.
4. **Table 4**: keyposes (K-means on AMASS, K = 128) and jitter as per-frame start strategies.
5. **Fig. 3 / 4**: arm to target, with and without the box, as figures/GIFs.
6. Optional: other methods of Table 1, Table 3.

## Commands

```bash
PYTHONPATH=/path/to/4D-Humans python scripts/hmr2_3dpw.py --pw3d data/3DPW --out results/hmr2_3dpw_test.npz
```

```bash
python scripts/eval_3dpw.py --smpl data/body_models/smpl --pw3d data/3DPW --pred results/hmr2_3dpw_test.npz
```

## Notes on the protocol

- Ground truth: gendered SMPL, global orientation rotated into the camera frame, `campose_valid` frames only.
- Joints: 14 LSP joints from `J_regressor_h36m.npy` if available (standard), otherwise from 4D-Humans' `SMPL_to_J19.pkl` (downloaded with the HMR2.0 checkpoint); the script prints which one it used.
- Boxes for HMR2.0: from the 3DPW 2D keypoints (fallback: projected 3D joints), cropped as in the HMR2.0 demo. HMR2.0's own evaluation uses preprocessed crops, so small differences to the paper's HMR2.0 row are expected.
