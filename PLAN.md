# Overnight plan: a working CLOAF by 12:00

Goal for 12:00, matching the first task of the project description ("run CLOAF on a small selection of SMPL motion sequences and examine the results as animations"):

1. CLOAF running on the real SMPL body, checked against the paper (Fig. 5 numbers).
2. A motion field network trained on an AMASS subset.
3. Results on 3 motion sequences with injected self-intersections: collisions before/after, distance to the clean motion, smoothness, for `--mode all` and `--mode colliding_only`.
4. Animations or screenshots of good and bad cases.
5. A short write-up (`docs/report_template.md`).

**Out of scope for tonight:** the 3DPW tables (they need HMR2.0 run on all of 3DPW). Write them down as next steps.

The times below count back from 12:00. If you start later, cut Phase 5 (second training block) first, never Phase 8 (the write-up).

---

## Phase 0: data access (1 h, the critical path)

Nothing else works without this.

- [ ] **SMPL.** On smpl.is.tue.mpg.de → Downloads, take **only** "Download version 1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs)" = `SMPL_python_v.1.1.0.zip`. Ignore the "Python 2.7" label: we only use the model files inside. Do not unzip or rename anything by hand; convert it (it removes the old `chumpy` dependency and sets the names smplx expects):
  ```bash
  python scripts/convert_smpl_pkl.py --src SMPL_python_v.1.1.0.zip --out data/body_models/smpl
  ```
  ```bash
  python tests/check_smpl.py --smpl data/body_models/smpl
  ```
- [ ] **AMASS.** Register at amass.is.tue.mpg.de (separate account). In the **Bodies** column click **"SMPL+H G"** for:
  - **ACCAD** (252 motions, 27 min): training
  - **SFU** (44 motions, 15 min): test sequences, never seen in training

  Skip the big sets (CMU, KIT, BMLrub, WEIZMANN) tonight.
- [ ] **Upload to Google Drive:**
  ```
  MyDrive/cloaf_data/SMPL_python_v.1.1.0.zip
  MyDrive/cloaf_data/amass/ACCAD.tar.bz2, SFU.tar.bz2
  ```
- [ ] **Local setup, while uploading.** Clone the repo, create a venv, `pip install -r requirements.txt`, then `python tests/smoke_test.py`. You need this local setup later to watch the animations.

**Gate:** the files are in Drive, the local smoke test prints `SMOKE TEST OK`, and `tests/check_smpl.py` prints PASS for all three genders.

## Phase 1: sanity checks on the real body (45 min)

Open `notebooks/colab_cloaf.ipynb` in Colab (File → Open notebook → GitHub → `arhannlp/cloaf_impl`) and set Runtime → GPU. Run:

- [ ] **Setup cells.**
- [ ] **Step 0.** Must print `SMOKE TEST OK`.
- [ ] **Step 1.** Copy both printed tables into the report. Expected: RE ≈ 0.07 at ‖dTheta‖ = 1e-2, RE ≈ 0.7 at 1e-1, and RE < 0.1 for S ≥ 100.
- [ ] **Step 2.** The hand-to-target distance must shrink towards 0.

**Gate:** the Step 1 numbers are in the same range as the paper. If RE is far off (for example > 0.5 at 1e-2), stop and send the output: the Jacobian or Θ is wrong, and training would be wasted.

## Phase 2: baselines and start of training (45 min)

- [ ] **Step 3.** Prepare AMASS and select the 3 test sequences.
- [ ] **Step 3b.** Collision test on clean AMASS: Col.Rate on clean mocap should be low with the default ("parts") test.
- [ ] **Step 4.** Training-free baselines, `point_target` and `smooth_target`, about 10 minutes. Results land in `cloaf_data/results/`.
  - **Check:** the line `corruption: ... N colliding vertices at the peak frame` must show N > 0. If N = 0, add `--corrupt_amp 1.6`.
- [ ] **Step 5.** Start training (120 minutes per run).
  - **First 20 minutes:** the loss must go down. If it is flat or `nan`, stop and see Troubleshooting below.

## Phase 3: training block 1 (2 h). Work on these in parallel locally

- [ ] Download `cloaf_data/results/*.npz` from Drive and watch the baselines:
  ```bash
  python scripts/view_results.py --smpl data/body_models/smpl --result <file>.npz
  ```
  Note what happens: does the arm still go into the torso? Does `smooth_target` stop it? What else moves?
- [ ] Re-read Sec. 3–4 of the paper next to `cloaf/integrate.py` and `cloaf/train.py`.
- [ ] Start the write-up: fill in Setup and Fig. 5 in `docs/report_template.md`.

## Phase 4: decision point (30 min)

When the training run stops, check the loss plot (Step 5, second cell) and the loss reference (third cell: network vs "zero field" = do nothing), then run **Step 6** with the current checkpoint and **Step 7**.

The network has learned something only if its loss is clearly below the zero field (for example 0.5x or less).

| What you see | Do |
|---|---|
| Network: fewer colliding frames after than before, and lower than `point_target` | Good. Continue training (Phase 5) |
| Loss went down, but the network is no better than `point_target` | Continue training once; if CMU is in Drive, set `TRAIN_SETS = ['ACCAD', 'CMU']` in Step 3, re-run the setup cells and Step 3, then continue training (`--resume` keeps the weights) |
| Loss did not go down | Troubleshooting below, then restart without `--resume` |
| Collisions *increase* after CLOAF | Send me the Step 6 output |

## Phase 5: training block 2 (2 h, optional; finish by 08:00)

- [ ] Re-run the training cell. It resumes from Drive.
- [ ] Meanwhile, try different `--sigma` values (0.02, 0.05) for `smooth_target` on one sequence. That is extra material for the extension discussion.

## Phase 6: final runs (finish by 08:30)

- [ ] **Step 6** with the final checkpoint: 3 sequences × 2 modes.
- [ ] **Step 7:** copy the table (`cloaf_data/results/summary.csv`) into the report.

## Phase 7: animations (finish by 09:30)

- [ ] Watch, for each sequence, the `all` and `colliding_only` results side by side.
- [ ] Save 2–4 screenshots or short videos (aitviewer GUI → export):
  - one case where the collision is fixed well,
  - one where the motion changes in an undesirable way,
  - one transition in `colliding_only` mode (the frame where correction starts or stops).

## Phase 8: write-up (finish by 11:30)

- [ ] Fill in `docs/report_template.md`: what works, the numbers, what you observed, the limitations and the next steps.
- [ ] Commit it to the repo (do not commit `data/` or the SMPL files).

**11:30–12:00: buffer.**

---

## How to read the numbers (`run_on_sequence.py`)

| Line | Meaning | Good result |
|---|---|---|
| `frames with collisions: before X, after Y` | Frames with at least one vertex inside the body | Y much smaller than X (paper: 0) |
| `vertex shift vs input` | How much CLOAF moved the body, in colliding and in clean frames | Small on clean frames, moderate on colliding ones |
| `distance to clean mocap` | Distance to the motion before we injected the collision | CLOAF lower than the input |
| `jerk mean / max` | Sudden jumps; the max reveals jumps at transitions | CLOAF max close to the input max |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError ... SMPL_NEUTRAL.pkl` | Run `scripts/convert_smpl_pkl.py` (Phase 0); `--smpl` must point to the folder it wrote |
| `No module named 'chumpy'` | You are loading the original files; convert them with `scripts/convert_smpl_pkl.py` |
| `cannot convert chumpy field ...` | Send me the full message (the converter met an unexpected file layout) |
| `no AMASS .npz files` | The tar.bz2 was not extracted, or the path is wrong; check `find data/amass -name '*.npz'` |
| `CUDA out of memory` | `--batch 8` or `--S 500` |
| Loss `nan` | `--lr 3e-5`, restart without `--resume` (or delete the `.state` file) |
| Loss flat from the start | `--lr 3e-4`; or `--max_gap 2` (smaller steps) |
| `0 colliding vertices at the peak frame` | `--corrupt_amp 1.6`, or another sequence |
| `warning: no collision-free start pose` | Fine occasionally; if most frames, the network is not trained enough |
| Run on one sequence takes too long | `--frames 60`; keep `--solver rk4` |
| Colab: "cannot connect to GPU backend" | GPU quota reached: use Kaggle Notebooks (free GPU hours per week); upload the data as a private Kaggle dataset and change `DATA` |
