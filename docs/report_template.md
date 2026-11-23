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

## 3. Collision detection on clean motion

The paper does not say how self-intersections are counted. On 200 random clean AMASS (ACCAD) frames:

| Test | @0 | @10 | @20 | @50 | @100 | Mean colliding vertices |
|---|---|---|---|---|---|---|
| Full mesh (vertex inside the body), rest-pose vertices ignored | 100 % | 93.5 % | 90.5 % | 80.5 % | 66.0 % | 149.8 |
| Full mesh, + crotch/armpit ignored | 100 % | 93.0 % | 87.5 % | 70.5 % | 32.5 % | 96.4 |
| Non-adjacent parts only, rest-pose vertices ignored | 38.5 % | 19.5 % | 14.5 % | 8.0 % | 4.5 % | 22.5 |
| **Non-adjacent parts only, + crotch/armpit ignored (used)** | **30.0 %** | **15.0 %** | **14.5 %** | **7.5 %** | **4.5 %** | **21.2** |

- Counting every vertex inside the body flags every clean frame: adjacent parts (upper arm against torso, forearm against upper arm) always overlap slightly in SMPL. We therefore count only vertices inside a non-adjacent body part, as SMPLify-X / TUCH do.
- The remaining hits on clean mocap are mostly hands (r_hand 2054, l_hand 1519 vertices over 200 frames): hands resting on the body. AMASS finger poses are dropped when converting SMPL+H to SMPL, so the hands are flat and sink slightly into the surface they touch.
- Col@0 is therefore very sensitive to light contact: 30 % on clean mocap, the same order as the paper's 39.2 % for HMR2.0 on 3DPW. Deep penetrations (Col@100) are rare on clean mocap (4.5 %).
- For sequence correction we treat a frame as colliding only with more than 50 colliding vertices (`collision_min`): this flags 7.5 % of clean frames instead of 30 %, while the injected collisions reach 500+ vertices.

## 4. Training

- Data: AMASS ACCAD, ~48k frames at 30 fps; test sequences from SFU (not used for training)
- Hardware: Colab T4 GPU. float32 with forward-mode AD for the Jacobian (the paper uses float64; on a T4 float64 was 5x slower: 6.6 vs 1.34 s/iteration)
- Settings the paper does not give, chosen here: hidden width 256, LeakyReLU, per-point network, batch 16, M = 10 Euler steps, frame gap 1–4
- Learning rate: 1e-4 for iterations 0–2000, 3e-4 for 2000–4000 (loss first rose, then fell below the earlier level), then back to 1e-4
- Loss curve: (insert plot)

Loss compared with two reference fields on the same pose pairs (`scripts/loss_reference.py`):

| Field | Loss after 2000 iterations | vs zero field |
|---|---|---|
| Zero field (body does not move) | 9.9e-4 | 1.00x |
| Network, 2000 iterations | 5.5e-4 | 0.56x |
| **Network, 4000 iterations** | **4.9e-4** | **0.50x** |
| Point target (each point straight to its target) | 5.9e-5 | 0.06x |
| Network, end of training | | |

## 5. Results on motion sequences with injected self-intersections

Collisions were injected by smoothly rotating one shoulder into the torso over 30 frames of clean AMASS motion, so the clean motion is known.

(paste `results/summary_runs.csv`)

| Sequence | Field | Mode | Colliding frames before → after | Dist. to clean (mm) input → CLOAF | Max jerk input → CLOAF |
|---|---|---|---|---|---|
| | point_target | all | | | |
| | smooth_target | all | | | |
| | network | all | | | |
| | network | colliding_only | | | |

## 6. Observations from the animations

- Which intersections are corrected:
- Where the movement changes in an undesirable way:
- `all` vs `colliding_only`: what happens at the transitions:
- (screenshots)

## 7. Limitations and open questions

- The network was trained for hours on one AMASS subset; the paper trained on all of AMASS on an A100.
- The guarantee is approximate: velocities are projected onto SMPL with J^† (least squares).
- Details the paper does not specify (see README, "Ambiguities"): collision detection, network size, training settings, crotch/armpit mask.

## 8. Next steps

1. Run HMR2.0 on 3DPW-test; calibrate the collision detector against Table 1 (HMR2.0 Col@0 = 39.2 %).
2. CLOAF on 3DPW: reproduce Table 2 (82.3 / 53.2 / 9.4 / 0.0 %) and Table 4.
3. Extension: systematic comparison of `all` vs `colliding_only`, focusing on transitions.
