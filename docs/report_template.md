# CLOAF re-implementation: first results

*Date:*  ·  *Repo:* https://github.com/arhannlp/cloaf_impl (commit: )

## 1. What was done

- Re-implemented CLOAF (Davydov et al., CVPR 2024) from the paper; the official code was not available.
- Components: SMPL Jacobian and pseudo-inverse (Eq. 3–5), motion field network (Eq. 8), ODE integration (Eq. 6), training with the trajectory loss (Eq. 9), self-intersection detection (generalized winding number), sequence correction (successive frames, Sec. 4.2).
- Extension started: correcting all frames vs only colliding frames.

## 2. Check against the paper: Jacobian linearity (Fig. 5, Fig. B.7/B.8)

| ‖δΘ‖ (S = 6890) | RE paper | RE ours |
|---|---|---|
| 1e-2 | ≈ 7.1e-2 | |
| 1e-1 | ≈ 7.0e-1 | |

| S (‖δΘ‖ = 1e-2) | RE paper | RE ours | Time ours (ms, GPU: ) |
|---|---|---|---|
| 10 | ≈ 0.7 | | |
| 100 | ≈ 0.05–0.08 | | |
| 1000 | < 0.1 | | |
| 6890 | < 0.1 | | |

## 3. Training

- Data: AMASS subset ______, ____ frames at 30 fps
- Hardware and time: ______ GPU, ____ h, ____ iterations
- Loss curve: (insert plot)
- Settings that the paper does not give and I chose: hidden width 256, LeakyReLU, per-point network, lr 1e-4, batch 16, M = 10, frame gap 1–4.

## 4. Results on motion sequences with injected self-intersections

Collisions were injected by smoothly rotating one shoulder into the torso over 30 frames of clean AMASS motion, so the clean motion is known.

(paste `results/summary.csv`)

| Sequence | Field | Mode | Colliding frames before → after | Dist. to clean (mm) input → CLOAF | Max jerk input → CLOAF |
|---|---|---|---|---|---|
| | point_target | all | | | |
| | smooth_target | all | | | |
| | network | all | | | |
| | network | colliding_only | | | |

## 5. Observations from the animations

- Which intersections are corrected:
- Where the movement changes in an undesirable way:
- `all` vs `colliding_only`: what happens at the transitions:
- (screenshots)

## 6. Limitations and open questions

- The network was trained for hours on one AMASS subset; the paper trained on all of AMASS on an A100.
- The guarantee is approximate: velocities are projected onto SMPL with J^† (least squares).
- Details the paper does not specify (see README, "Ambiguities"): collision detection, network size, training settings, crotch/armpit mask.

## 7. Next steps

1. Run HMR2.0 on 3DPW-test; calibrate the collision detector against Table 1 (HMR2.0 Col@0 = 39.2 %).
2. CLOAF on 3DPW: reproduce Table 2 (82.3 / 53.2 / 9.4 / 0.0 %) and Table 4.
3. Extension: systematic comparison of `all` vs `colliding_only`, focusing on transitions.
