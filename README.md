# CLOAF re-implementation

Project: *Correcting Self Intersections in SMPL*
Paper: Davydov et al., *CLOAF: CoLlisiOn-Aware Human Flow*, CVPR 2024 ([arXiv 2403.09050](https://arxiv.org/abs/2403.09050))

The official code (github.com/cvlab-epfl/CLOAF) was not available, so this is a re-implementation from the paper.

---

## Current status: what works and what does not

| Part | Status | Needs |
|---|---|---|
| SMPL wrapper, surface sampling, Jacobian, J^† (`body.py`) | Works; tested on a fake mesh | SMPL files to test on the real body |
| Self-intersection detector (`collision.py`) | Works; tested on a fake mesh | Calibration against Table 1 |
| Simple hand-crafted field (Sec. 4.4) | Works; **no training needed** | SMPL files |
| Learned motion field `f_ω` (`model.py`) | Code runs, but the network is **untrained** | Training on AMASS (GPU) |
| Pose correction (`correct_sequence`) | Code runs, but the output is **meaningless until `f_ω` is trained** | A trained checkpoint |
| 3DPW evaluation (`eval_3dpw.py`) | Draft | 3DPW, HMR2.0 predictions, `J_regressor_h36m.npy` |

**In short:** if you give it SMPL poses today, it will output SMPL parameters, but they will not be corrected poses. The network's weights are random. CLOAF only works after Step 4 (training).

---

## Setup

```bash
git clone https://github.com/arhannlp/cloaf_impl.git
```

```bash
cd cloaf_impl
```

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

On Windows, activate with `.venv\Scripts\activate` instead.

```bash
pip install -r requirements.txt
```

Data (all academic licenses, register yourself; never commit them, `.gitignore` excludes `data/`):

- **SMPL** (smpl.is.tue.mpg.de): `data/body_models/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl`
- **AMASS** (amass.is.tue.mpg.de): training data, SMPL+H format
- **3DPW** (virtualhumans.mpi-inf.mpg.de/3DPW): evaluation
- **4D-Humans / HMR2.0** (github.com/shubham-goel/4D-Humans): baseline predictions
- **J_regressor_h36m.npy**: from the SPIN data package, for the standard 14-joint 3DPW metric

---

## Testing guide, step by step

Each step only uses the parts checked in the previous steps. Stop at the first step that fails and fix it before moving on.

### Step 0: Smoke test (no data needed)

```bash
python tests/smoke_test.py
```

It builds a fake ellipsoid body and runs every part of the code once. Expected output: seven `[PASS]` lines and `SMOKE TEST OK`.

| Check | What it verifies | Paper |
|---|---|---|
| 1 | Two ways of computing the Jacobian agree (difference ~1e-10) | Eq. 3 |
| 2 | J·dΘ predicts how points move for a small pose change | Eq. 3, Fig. 5 |
| 3 | The pseudo-inverse recovers dΘ from the point motion | Eq. 5, Eq. 12 |
| 4 | Winding number ≈ 1 inside the body, ≈ 0.5 on the surface | Sec. 4.1 |
| 5 | The training loss is finite and gradients reach the network | Eq. 9 |
| 6 | The ODE solver runs (distance to target is meaningless: untrained) | Eq. 6 |
| 7 | Sequence correction runs end to end | Sec. 4.2 |

Read `tests/smoke_test.py` next to its output: each block is the shortest example of how to call one part of the code.

### Step 1: Jacobian on the real SMPL (needs SMPL files)

```bash
python scripts/check_linearity.py --smpl data/body_models/smpl --device cpu
```

Prints two tables. Compare them with the paper:

- `[Fig. B.7]`: RE around 7e-2 at ||dTheta|| = 1e-2 and around 7e-1 at 1e-1 (Fig. B.8)
- `[Fig. 5]`: RE around 0.7 for S = 10, below 0.1 for S ≥ 100

If this does not match, the problem is in the Jacobian or in the definition of Θ. Do not continue until it does.

### Step 2: Integration with a simple field (needs SMPL files)

```bash
python scripts/demo_target_field.py --smpl data/body_models/smpl --device cpu
```

```bash
python scripts/demo_target_field.py --smpl data/body_models/smpl --device cpu --box
```

Without `--box`, the printed hand-to-target distance should shrink towards 0. With `--box`, it should stop at the box surface (Fig. 3 and Fig. 4). Watch `results/target_field.npz` in aitviewer to see the motion.

### Step 3: Calibrate the collision detector (needs 3DPW and HMR2.0 outputs)

```bash
python scripts/eval_3dpw.py --smpl data/body_models/smpl --pw3d data/3DPW --pred data/hmr2_3dpw_test.npz --no_cloaf
```

Target: the HMR2.0 row of Table 1/2 (Col.Rate@0 = 39.2 %, @100 = 21.6 %). The paper does not explain how collisions are detected, so tune the threshold and excluded regions here, **before** evaluating CLOAF.

### Step 4: Train `f_ω` (needs AMASS and a GPU)

Convert AMASS with `prepare_amass()` in `cloaf/train.py`, then:

```bash
python -m cloaf.train --smpl data/body_models/smpl --amass data/amass_smpl_30fps.npz
```

Start by overfitting a small subset (for example CMU only): the loss must go down. Then train on everything.

### Step 5: CLOAF on 3DPW

Run `eval_3dpw.py` with `--field checkpoints/field.pt --mode all`. Target: the CLOAF row of Table 2, and Table 4.

### Step 6: Extension (from the project description)

Compare `--mode colliding_only` with `--mode all`:

- Col.Rate
- MPJPE and Accel
- `temporal_jerk` at the transitions between corrected and uncorrected frames
- side-by-side videos in aitviewer

---

## Where to run what

| Task | Where | Why |
|---|---|---|
| Writing code | Locally, VS Code + git | Modules are easy to test and move to a server |
| Steps 0–2 | Local CPU is enough | No training |
| Training and 3DPW evaluation | TUM GPU server (chair server or LRZ AI Systems; ask Tony) | The paper used an A100; the Jacobian calls SMPL in float64 many times |
| Watching results | Locally, with aitviewer or Jupyter | Needs a screen |
| Google Colab | Fallback only | Sessions disconnect; licensed files must be uploaded |

---

## Code map (in the order of the paper)

```
cloaf/
  body.py         SMPL wrapper, surface sampling (S=1000, crotch/armpits excluded),
                  X(Θ), Jacobian (finite differences or jacfwd, float64), J^† f   [Sec. 3, Eq. 3-5]
  model.py        f_ω: Fourier features (n_f=20) + 6-layer concat-MLP (ShapeFlow style),
                  inputs (x, Θ(t), Θ1, Δt), output scaled by expected speed        [Sec. 3, Eq. 8]
  integrate.py    ODE (Eq. 6) + dopri5; simple target field (Eq. 10-11, C.14);
                  sequence correction: mode="all" (paper) / "colliding_only" (extension); Jitter
  train.py        AMASS preparation, pair sampling, trajectory loss (Eq. 9, linear Θ~)
  collision.py    Generalized Winding Number self-intersection test, Col.Rate@C   [Sec. 4.1]
  metrics.py      MPJPE, PA-MPJPE, Accel.Err, temporal jerk                         [Sec. 4.1]
scripts/
  check_linearity.py   Step 1: Fig. 5 and B.7/B.8 (no training)
  demo_target_field.py Step 2: Fig. 3/4 (no training)
  eval_3dpw.py         Steps 3 and 5: Table 1, 2, 4
tests/smoke_test.py    Step 0: end-to-end test with a fake model
```

Suggested reading order: Sec. 3 of the paper → `body.py` (run Step 1) → `integrate.py` and Sec. 4.4 (run Step 2) → `model.py`, `train.py` → `collision.py`, `metrics.py`, `correct_sequence()`, `eval_3dpw.py`.

---

## Validation table (show this to the supervisor)

| Experiment | Paper value | Source | Ours |
|---|---|---|---|
| RE, ‖δΘ‖=1e-2, S=6890 | ≈ 7.1e-2 | Fig. B.8 | |
| RE, ‖δΘ‖=1e-1, S=6890 | ≈ 7.0e-1 | Fig. B.8 | |
| RE, S=10 / S≥100 / S=1000 | ≈0.7 / ≈0.05–0.08 / <0.1 | Fig. 5 | |
| Jacobian time, S=1000 / 6890 | ~15–20 ms / ~320 ms (A100) | Fig. 5 | |
| HMR2.0 Col@0 / @100 | 39.2 % / 21.6 % | Table 1 | |
| PARE, TUCH, EFT, SPIN Col@0 | 36.1 / 23.7 / 17.8 / 15.5 % | Table 1 | (optional) |
| HMR2.0 MPJPE / P-MPJPE / Accel | 82.0 / 52.7 / 16.1 | Table 2 | |
| **CLOAF** MPJPE / P-MPJPE / Accel / Col@0 | **82.3 / 53.2 / 9.4 / 0.0 %** | Table 2 | |
| CLOAF-Jitter | 81.6 / 52.6 / 15.9 | Table 4 | |
| CLOAF-Keyposes (K=128) | 82.9 / 53.6 / 16.4 | Table 4 | |
| EFT + CLOAF fine-tune (diff) | P-MPJPE 55.4, Col 3.4 % | Table 3 | (probably out of scope) |

### Ambiguities and inconsistencies in the paper

Report these as assumptions and ask the supervisor:

1. **HMR2.0 P-MPJPE differs between tables:** 54.3 in Table 1, 52.7 in Table 2. Table 1 probably quotes the HMR2.0 paper; Table 2 is the authors' own run.
2. **How self-intersections are detected:** the paper only says "more than C vertices inside the body". We use GWN > 1.0.
3. **Column "@100 (w/o col.)" in Table 1** is not clearly defined.
4. **Network architecture:** hidden width, activation, whether the network is per-point, and the Fourier frequency range are not given. See the assumptions in `model.py`.
5. **Crotch and armpit mask:** only "excluded following [26]". `body.py` uses a simple radius mask; check it visually.
6. **AMASS training details:** frame gap ("consecutive poses, Δt_max=1"), fps, betas (SMPL+H to SMPL), learning rate, iterations, batch size and M are not given.
7. **Shape β and global orient:** Θ is 69-dim (Fig. B.8). β and global orient are taken from the target frame. Θ0 may collide with a different β, so `correct_sequence` re-checks it.
8. **Dopri5 tolerances** and how many runs the numbers are averaged over are not given.
9. **First frame:** what happens when it collides is unclear. We run Jitter first, then CLOAF.

Also worth discussing: the non-intersection guarantee holds for the free flow of points. CLOAF projects the velocities onto SMPL with J^†, which is a least-squares approximation, so in practice the guarantee is approximate. The paper shows 0.0 % empirically.

---

## Known gaps in the draft (TODO)

- `eval_3dpw.py`: Accel should be computed per contiguous block of `campose_valid` frames. A converter that writes HMR2.0 outputs to the expected npz is still needed.
- Keypose strategy (K-means, K=128) is a skeleton; the paper uses keypoint distance.
- Differentiable fine-tuning (Table 3) would need `odeint_adjoint`; probably out of scope.
- Compare `jacobian(method="jacfwd")` with `fd` on the real SMPL and keep the faster one.
- Contact the authors (Andrey Davydov, EPFL) for code or a checkpoint.
