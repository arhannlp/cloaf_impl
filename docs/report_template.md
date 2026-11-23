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

Collisions were injected by smoothly rotating one shoulder into the torso over 30 frames (frames 30–59) of clean AMASS motion, so the clean motion is known. Test sequence: SFU `0005_2FeetJump001` (held out from training), 90 frames at 30 fps, injection +1.2 rad on `body_pose[50]` (295 colliding vertices at the peak frame). A frame counts as colliding with more than 50 colliding vertices. Solver: RK4, 10 steps. Network after 4000 iterations.

| Field | Mode | Colliding frames before → after | Colliding vertices (sum) before → after | Dist. to clean motion (mm) input → CLOAF | Max jerk (mm) input → CLOAF | s/frame |
|---|---|---|---|---|---|---|
| Network | all | 49 → 14 | 15811 → 6914 | 14.9 → 524.3 | 18.1 → 1998.6 | 2.57 |
| Network | colliding_only | 49 → 38 | 15811 → 21712 | 14.9 → 207.4 | 18.1 → 2451.1 | 1.51 |
| smooth_target (σ = 3 cm) | all | 49 → 46 | 15811 → 16249 | 14.9 → 15.8 | 18.1 → 18.0 | 2.60 |
| point_target (Supp. C) | all | 49 → 48 | 15811 → 16169 | 14.9 → 16.0 | 18.1 → 17.7 | 2.59 |

Start poses used (network, all): previous frame 75, last collision-free pose 11, skipped 4 (frames 0–3 already collide in the clean motion and no collision-free start was found).

### What the numbers say

1. **The network removes collisions only by destroying the pose.** In `all` mode the colliding frames drop from 49 to 14, but the body moves on average half a metre away from the clean motion and jumps by up to 2 m between frames. This is not a correction.
   Likely causes: (a) the network is under-trained (loss 0.50x of a zero field vs 0.06x for the point-target reference); (b) the test situations are outside the training distribution: training pairs are at most 4 frames apart and clean, while here the start pose can be the last collision-free frame many frames back and the target has a shoulder bent by 1.2 rad; (c) with successive frames, every error is carried into the next frame.
2. **The training-free fields do not remove collisions.** They move the body by only a few millimetres: each point is driven to its position in the target pose, so the integration ends at the colliding target. The non-intersection argument holds for points that move freely; after the least-squares projection onto SMPL (J^†) the torso cannot "dent" where the arm pushes, and the arm ends up inside it anyway. The same effect appeared in the simple target-field test (Sec. 2: 0 → 7 colliding vertices).
3. **The clean motion itself collides.** 19 frames outside the injected window exceed 50 colliding vertices (hands close to the body during the jump), and the first 4 frames have no collision-free start pose. This sequence is harder than expected as a test case.

## 5b. Where the guarantee breaks: free flow vs projected flow

`scripts/free_flow_test.py` integrates the same field twice from the last collision-free frame (29, 10 colliding vertices) towards three colliding frames of the same sequence: once with all 6890 vertices moving freely (no SMPL, as in Eq. 1), once projected onto SMPL with J^+ (Eq. 6). RK4, 20 steps.

| Field | Target frame | Colliding vertices in target | Free flow: at end (dist. to target) | Projected flow: at end (dist. to target) | Projection error |
|---|---|---|---|---|---|
| smooth_target (σ = 3 cm) | 35 | 651 | **22** (3.9 mm) | 562 (3.4 mm) | 16.7 % |
| smooth_target | 45 | 296 | **128** (26.0 mm) | 334 (11.5 mm) | 31.1 % |
| smooth_target | 55 | 923 | **20** (6.6 mm) | 918 (5.1 mm) | 17.9 % |
| point_target (control) | 35 / 45 / 55 | 651 / 296 / 923 | 619 / 309 / 927 | 619 / 311 / 928 | 3–12 % |
| network (4000 it.) | 35 / 45 / 55 | 651 / 296 / 923 | 17 / 111 / 33 (113 / 379 / 128 mm) | 0 / 639 / 0 (81 / 359 / 245 mm) | 68–77 % |

Projection error = ||J J^+ f − f|| / ||f||, the share of the requested point motion that SMPL cannot follow.

- **The flow itself works.** With a field that is a function of position (smooth_target), freely moving points avoid 57–98 % of the collisions while staying within a few millimetres of the target: trajectories cannot cross, as Eq. 1 promises. This also confirms that the collision test behaves as intended.
- **The projection onto SMPL undoes it.** Projected with J^+, the same field reaches almost all of the target's collisions. SMPL cannot follow 17–31 % of the requested motion, and what it drops is the shared motion of the few points at the contact: they are a small fraction of the 1000 samples, so the least-squares fit ignores them.
- The control behaves as expected: point_target is not a function of position (each point has its own target), so it keeps the collisions with or without projection.
- The network asks for motions SMPL cannot perform (68–77 % projection error) and does not reach the target even in free flow: it is under-trained, independently of the projection issue.

**Consequence.** The paper's guarantee holds for the free flow; after the projection it is only approximate, and in our tests it is lost. The paper reports 0.0 % collisions, so either its learned field asks only for motions SMPL can follow at contacts, or its collision measure differs; worth asking the authors.

**Extension 1: weighted projection.** Give points near a non-adjacent body part more weight in the least squares, dΘ = (J^T W J)^-1 J^T W f with w = 1 + λ·exp(−d²/2τ²) (d: distance to the nearest point of a non-adjacent part, τ = 2 cm; λ = 0 is the paper). smooth_target field, projected flow, colliding vertices at the end (distance to target):

| λ | Frame 35 (target 651) | Frame 45 (target 296) | Frame 55 (target 923) | Projection error |
|---|---|---|---|---|
| 0 (paper) | 562 (3.4 mm) | 334 (11.5 mm) | 918 (5.1 mm) | 17–31 % |
| 20 | 487 (6.0 mm) | 436 (21.0 mm) | 896 (9.5 mm) | 27–43 % |
| 100 | 385 (10.5 mm) | 431 (31.8 mm) | 787 (15.9 mm) | 44–64 % |
| 500 | 318 (15.8 mm) | 445 (48.7 mm) | 686 (23.7 mm) | 60–109 % |
| free flow (reference) | 22 | 128 | 20 | – |

Weighting helps for two of the three frames (−26 % to −51 %), hurts the third, and stays far from the free flow. The reason is the requested motion itself: in the free flow the torso surface yields where the arm pushes it, and SMPL has no parameter that dents the torso. However strongly the contact points are weighted, SMPL can only follow an average of their request, and the arm still moves into the torso, only more slowly.

**Extension 2: contact constraint.** Instead of re-weighting the request, forbid the part of the motion that causes the collision: for every pair of sample points on non-adjacent parts closer than 5 cm, their distance may not decrease, n^T (J_i − J_j) dΘ ≥ 0 with n the unit vector between them. Violated pairs are enforced with a large penalty in a few active-set iterations (`body.contact_constrained_solve`); sliding along the contact remains allowed, so SMPL can stop the arm at the torso instead of denting it. Checked on a synthetic case (all approaching pairs stopped). Result on the real body (smooth_target, frame 45, RK4 10 steps, CPU):

| Frame 45 (target: 296 colliding vertices) | Colliding at the end | Worst along the path | Distance to target | Projection error |
|---|---|---|---|---|
| Free flow (reference) | 153 | 153 | 26.8 mm | – |
| Projected, J^+ (paper) | 328 | 936 | 11.6 mm | 30.6 % |
| **Projected, contact constraint 5 cm** | **17** | **30** | 115.9 mm | 89.4 % |

The start pose already has 10 colliding vertices, so the constraint creates almost no new collision, along the whole path (38 active constraints on average). It even beats the free flow, where the torso yields to the arm: here the arm is stopped at the contact instead. The price is the distance to the colliding target (116 mm on average over all vertices). Some distance is expected, because the target has the arm 1.2 rad inside the torso, but the average suggests that other body parts move as well to satisfy the constraints; the distance to the clean motion and the animations will tell whether this is a good correction.

## 5c. Sequence correction with the contact constraint

Same sequence (SFU `0005_2FeetJump001`), 60 frames, collision injected in frames 20–39 (`body_pose[50]` +1.2 rad, 319 colliding vertices at the peak), smooth_target field, `colliding_only` mode (frames with ≤ 50 colliding vertices are kept), RK4 10 steps, Colab CPU.

| | Input (with injected collision) | Contact constraint 5 cm | No constraint (paper's J^+) |
|---|---|---|---|
| Colliding frames | 28 / 60 | **3 / 60** (frames 0–2, no collision-free start) | |
| Colliding vertices (sum) | 12108 | **2002** | |
| Distance to the clean motion | 15.2 mm | **11.9 mm** | |
| Jerk mean / max | 5.0 / 17.3 mm | 12.4 / 63.0 mm | |
| Shift of corrected frames | – | 42.0 mm | |
| Start pose of corrected frames | – | previous output 25, fallback 0, skipped 3 | |
| Time | – | 46 s/frame (CPU) | |

- **The collisions are gone** except in the first three frames, where the clean motion itself already collides and no collision-free start exists.
- **The corrected motion is closer to the clean motion than the input** (11.9 vs 15.2 mm). The 116 mm "distance to target" of the single-frame test (Sec. 5b) was misleading: the target is the wrong pose; what matters is the distance to the clean motion.
- **The chain of successive frames holds**: every corrected frame could start from the previous output, because every output was collision-free. Without the constraint, about half of the frames had to fall back to an older pose.
- **The motion is less smooth**: the largest frame-to-frame jerk grows from 17 to 63 mm. Likely at the transitions between corrected and kept frames (start and end of the collision window), which is exactly the trade-off the project description asks about. To check: the animation around frames 20 and 39, and the same run in `all` mode.

## 6. Observations from the animations

(watch `*_net_all.npz`, `*_net_colliding_only.npz`, `*_smooth_target_all.npz` with `scripts/view_results.py`)

- Which intersections are corrected:
- Where the movement changes in an undesirable way:
- `all` vs `colliding_only`: what happens at the transitions:
- (screenshots)

## 7. Limitations and open questions

- Training: about 1.5 h on a Colab T4 (4000 iterations, ACCAD only) in float32. The paper trained on all of AMASS on an A100 in float64; training length is not reported.
- Training distribution: pose pairs 1–4 frames apart from clean mocap. Correction needs larger gaps and colliding targets; the paper does not say whether it trains on such pairs.
- The non-intersection guarantee is only approximate after projecting the velocities onto SMPL with J^† (least squares); in our tests this is enough to leave the collisions in place.
- The contact constraint stops any further approach of point pairs on non-adjacent parts once they are closer than 5 cm. Existing contacts can stay, slide or separate, but in corrected frames a new contact (e.g. a hand moving onto a knee) stops about 5 cm short. Possible refinement: allow approach in proportion to the remaining gap, approach rate ≤ κ (d − d_min) with d_min ≈ 5 mm, and sample more densely near contacts (1000 samples are ~4 cm apart on the body surface).
- Details the paper does not specify (see README, "Ambiguities"): collision detection, network size and inputs, training settings, crotch/armpit mask, solver tolerances.

## 8. Next steps

1. Train longer and on more data (all of AMASS, ideally on a university GPU), and with larger frame gaps so that the network sees start/target pairs like those met during correction; check whether the loss reaches the point-target level.
2. Contact the authors for code or a checkpoint, to separate implementation issues from training issues.
3. Run HMR2.0 on 3DPW-test; calibrate the collision detector against Table 1 (HMR2.0 Col@0 = 39.2 %).
4. CLOAF on 3DPW: reproduce Table 2 (82.3 / 53.2 / 9.4 / 0.0 %) and Table 4.
5. Extension: systematic comparison of `all` vs `colliding_only`, focusing on transitions, once the field works.
