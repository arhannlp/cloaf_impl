# Plan: a working CLOAF

Goal: a CLOAF implementation that removes self-intersections from SMPL motion sequences while keeping the motion natural. Reproducing the paper's numbers is secondary.

## Done

- [x] Core method: SMPL Jacobian and pseudo-inverse (checked against Fig. 5 / B.7), flow in the SMPL parameters (Eq. 6), RK4 integration, successive frames (Sec. 4.2), jitter start (Sec. 4.5).
- [x] Self-intersection test that works on real motion (non-adjacent body parts, rest-pose and crotch/armpit vertices ignored, > 50 vertices per frame).
- [x] Diagnosis: the flow avoids collisions when points move freely, the projection onto SMPL (J^+) brings them back (free-flow test).
- [x] Fix: contact-constrained projection: removes ~85 % of the colliding vertices on a held-out sequence, closer to the clean motion than the input.
- [x] Smooth sequences: return phase (no jump at the end of a corrected stretch) and temporal smoothing (no jitter inside it).
- [x] Rest-pose start for sequences that begin in collision.
- [x] One command (`scripts/cloaf_correct.py`), GIF rendering (`scripts/render_gif.py`), Colab cell.

## Next

1. **Generalisation** (GPU, ~20 min): the four methods on two more SFU sequences (paper / all / colliding_only / final), tables and per-frame plots.
2. **Animations** (GPU, a few minutes each): GIFs of the final method on 2–3 sequences; describe in the report what is corrected and where the motion changes.
3. **Report and supervisor e-mail.**

## Later, with a proper GPU (the 3-week window)

4. **Learned field** (the paper's network): train on several AMASS sets with frame gaps 1–30, a learning-rate schedule and validation (`scripts/loss_reference.py`, `scripts/free_flow_test.py`). Compare network + contact constraint with smooth_target + contact constraint; make the better one the default (`--field`).
5. **Contact margin**: let pairs approach down to a small gap (approach rate ≤ κ (d − d_min)) so that new contacts, such as a hand moving onto a knee, can form.
6. **Real estimator output**: HMR2.0 on 3DPW, Col.Rate and MPJPE before/after (Table 1/2 of the paper).

## Quick reference

```bash
python scripts/cloaf_correct.py --smpl data/body_models/smpl --input motion.npz --output motion_cloaf.npz
```

```bash
python scripts/render_gif.py --smpl data/body_models/smpl --result motion_cloaf.npz
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError ... SMPL_NEUTRAL.pkl` | Run `scripts/convert_smpl_pkl.py` on the SMPL zip; `--smpl` points to the folder it wrote |
| `No module named 'chumpy'` | Same: convert the original SMPL files |
| Very slow (tens of seconds per frame) | No GPU: Colab Runtime → Change runtime type → T4 GPU |
| Colab: "cannot connect to GPU backend" | GPU quota used up: wait a few hours, or use Kaggle Notebooks |
| `0 colliding vertices at the peak frame` (injected tests) | `--corrupt_amp 1.6` or another sequence |
| `warning: no collision-free start pose` | Rare now (rest-pose fallback); the frame is left unchanged |
