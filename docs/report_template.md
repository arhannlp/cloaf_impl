# CLOAF re-implementation: first results

*Date:*  ·  *Repo:* https://github.com/arhannlp/cloaf_impl (commit: )

## 1. What was done

- Re-implemented CLOAF (Davydov et al., CVPR 2024) from the paper; the official code was not available.
- Components: SMPL Jacobian and pseudo-inverse (Eq. 3–5), motion field network (Eq. 8), ODE integration (Eq. 6), training with the trajectory loss (Eq. 9), self-intersection detection (generalized winding number), sequence correction (successive frames, Sec. 4.2).
- Extension started: correcting all frames vs only colliding frames.

## 2. Check against the paper: Jacobian linearity (Fig. 5, Fig. B.7/B.8)

| ‖δΘ‖ (S = 6890) | RE paper | RE ours |
|---|---|---|
| 1e-6 | ≲ 1e-5 | 2.6e-6 |
| 1e-4 | ≲ 1e-3 | 5.7e-4 |
| 1e-2 | ≈ 7.1e-2 | 2.5e-2 |
| 1e-1 | ≈ 7.0e-1 | 2.8e-1 |

‖f‖ grows linearly with ‖δΘ‖ (slope 1) from 1e-9 to 1e-2, as in Fig. B.7; linearity breaks down above ≈ 3e-2.
Our RE is about 3x lower than the paper's. Θ0 was sampled as 0.2·N(0, I) around the rest pose (the paper does not say how it samples Θ0), where SMPL is more linear; re-run with AMASS poses (`--amass`) to compare.

| S (‖δΘ‖ = 1e-2) | RE paper | RE ours | Time paper (ms, A100) | Time ours (ms, T4) |
|---|---|---|---|---|
| 10 | ≈ 0.7 | 0.81 | ~15 | 48 |
| 30 | | 0.11 | | 47 |
| 100 | ≈ 0.05–0.08 | 0.076 | ~15 | 47 |
| 300 | | 0.036 | | 47 |
| 1000 | < 0.1 | 0.033 | ~20 | 48 |
| 6890 | < 0.1 | 0.053 | ~320 | 48 |

The accuracy trend matches: few points give a poor estimate, S ≥ 100 is accurate, S = 1000 is a good choice.
Our time does not grow with S because we compute the Jacobian with central finite differences: every evaluation runs the full SMPL mesh, so the cost is independent of S. The paper's time grows with S, which suggests a per-output (reverse-mode) Jacobian.

### Real SMPL sanity checks (tests/check_smpl.py)

- All three models load (6890 vertices, 13776 faces); the mesh is watertight.
- The SMPL template already self-intersects in the rest pose at 41 vertices (33 inner thigh / crotch, 8 feet; winding number ≈ 1.45, so not a numerical artefact). This matches the paper's remark that the crotch is prone to natural self-intersections. These vertices are ignored when counting collisions.
- Fully bent elbows: 228 colliding vertices, so real collisions are detected.

### Simple target field without training (Sec. 4.4, Fig. 3)

Hand-to-target distance shrinks from 0.474 m to 0.225 m (T = 400; longer T would reach the target), so the ODE + Jacobian + J^† chain works.
Colliding vertices grow from 0 to 7 while the arm is raised. Only the hand region has a non-zero velocity; the rest of the arm follows through the least-squares projection J^†, and the armpit (excluded from sampling) has no points to constrain it. This is a concrete case where the non-intersection guarantee holds only approximately after the projection onto SMPL.

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
