"""
CLOAF integration (Eq. 6):   dΘ/dt = J^† f(X(Θ), t | Θ0, Θ1),   Θ(t0) = Θ0

- Learned field     : MotionField (model.py)
- Simple field (4.4): TargetField + Bézier blending (Eq. 10-11, Supp. C.14)
- Sequence correction: "successive frames" (Sec. 4.2) and, as an extension,
                       "colliding_only" (from the project proposal)
"""
from __future__ import annotations

import torch
from torchdiffeq import odeint

from .body import SMPLBody, contact_constrained_solve, pinv_solve
from .model import DT_MIN


# ====================================================================== #
#                         Right-hand side of the ODE                      #
# ====================================================================== #
class CLOAFODE(torch.nn.Module):
    """Velocity in Θ space: J^† f. Solved with torchdiffeq.odeint."""

    def __init__(self, body: SMPLBody, field, theta_1, sample, betas=None,
                 global_orient=None, t1: float = 1.0, jac_method="fd",
                 contact_weight: float = 0.0, contact_tau: float = 0.02,
                 contact_dist: float = 0.0, contact_mu: float = 1e3):
        super().__init__()
        self.body, self.field, self.sample = body, field, sample
        # Weighted projection (our extension, not in the paper): points close to a
        # non-adjacent body part get weight 1 + contact_weight * exp(-d^2 / 2 tau^2) in the
        # least-squares projection, so that SMPL follows the motion at contacts first.
        # contact_weight = 0 reproduces the paper's J^+.
        self.contact_weight, self.contact_tau = contact_weight, contact_tau
        # Contact constraint (our extension): points of non-adjacent parts closer than
        # contact_dist may not approach each other (see body.contact_constrained_solve).
        # contact_dist = 0 disables it.
        self.contact_dist, self.contact_mu = contact_dist, contact_mu
        self.n_active = []
        self.sample_part = None
        if contact_weight > 0 or contact_dist > 0:
            if body._part_test is None:
                raise ValueError("contact weights/constraints need SMPLBody(collision_mode='parts')")
            fidx = sample[0] if sample is not None else None
            verts = body.faces[fidx, 0] if fidx is not None else torch.arange(
                body.smpl.v_template.shape[0], device=body.device)
            self.sample_part = body._part_test.vertex_part[verts]              # (S,)
            self.far_pairs = ~body._part_test.near[self.sample_part][:, self.sample_part]  # (S,S)
        self.theta_1, self.betas, self.go, self.t1 = theta_1, betas, global_orient, t1
        self.jac_method = jac_method
        self.nfe = 0  # number of function evaluations (solver steps)
        self.proj_err = []  # ||J J^+ f - f|| / ||f||: share of the requested motion SMPL cannot follow
        with torch.no_grad():
            self.x_1 = body.points(theta_1, sample, betas, global_orient)

    def forward(self, t, theta):
        self.nfe += 1
        x = self.body.points(theta, self.sample, self.betas, self.go)            # [Sampling]
        J = self.body.jacobian(theta, self.sample, self.betas, self.go, method=self.jac_method)
        dt = (self.t1 - t).expand(theta.shape[0]).to(theta)
        f = self.field(x.float(), theta.float(), self.theta_1.float(), dt.float(),
                       x_target=self.x_1.float())                               # [Field velocities]
        f = f.reshape(f.shape[0], -1)
        if self.contact_dist > 0:
            dtheta, na = contact_constrained_solve(J, f, x, self.far_pairs, self.contact_dist,
                                                   self.contact_mu)
            self.n_active.append(sum(na) / len(na))
        else:
            w = self.contact_weights(x) if self.contact_weight > 0 else None
            dtheta = pinv_solve(J, f, weights=w)                                # [Parametric correction]
        with torch.no_grad():
            r = (J.double() @ dtheta[..., None])[..., 0] - f.double()
            self.proj_err.append((r.norm(dim=-1) / f.double().norm(dim=-1).clamp_min(1e-12)).mean().item())
        return dtheta


    @torch.no_grad()
    def contact_weights(self, x):
        """(B,S) weights: large for points near a non-adjacent body part."""
        d = torch.cdist(x.float(), x.float())                                  # (B,S,S)
        d = d.masked_fill(~self.far_pairs[None], float("inf")).min(-1).values  # nearest far point
        return 1.0 + self.contact_weight * torch.exp(-d ** 2 / (2 * self.contact_tau ** 2))


@torch.no_grad()
def cloaf_solve(body, field, theta_0, theta_1, S=1000, betas=None, global_orient=None,
                method="dopri5", rtol=1e-4, atol=1e-5, n_out=2, seed=0, options=None, **kw):
    """Integrate from Θ0 (collision-free) towards the target Θ1.
    method="dopri5" is the paper's adaptive solver. For speed, use a fixed-step
    solver instead, e.g. method="rk4", options=dict(step_size=0.1) (10 steps).
    Returns the trajectory (n_out, B, 69) and the number of solver evaluations."""
    g = torch.Generator(device=body.device).manual_seed(seed)
    sample = body.sample_surface(S, generator=g)
    ode = CLOAFODE(body, field, theta_1.to(body.dtype), sample, betas, global_orient, **kw)
    ts = torch.linspace(0, 1, n_out, device=body.device, dtype=body.dtype)
    traj = odeint(ode, theta_0.to(body.dtype), ts, method=method, rtol=rtol, atol=atol,
                  options=options)
    return traj, ode.nfe


# ====================================================================== #
#              Simple (hand-crafted) field — Sec. 4.4 / Supp. C            #
# ====================================================================== #
def bezier_blend(r, r_in, r_out, alphas=(0, 0, 0, 1, 1)):
    """Eq. 10: b(r) = Σ α_p B_p^4((r-r_in)/(r_out-r_in))."""
    from math import comb
    u = ((r - r_in) / (r_out - r_in)).clamp(0, 1)
    return sum(a * comb(4, p) * u ** p * (1 - u) ** (4 - p) for p, a in enumerate(alphas))


class PointTargetField(torch.nn.Module):
    """Training-free baseline: every sampled point moves straight towards its own
    position in the target pose, f_i = (x_i(Θ1) - x_i) / Δt.

    This is the "simple field" of Supp. C ("Simple Field vs NN"): integrating it is
    an interpolation in coordinate space that minimises ||x(Θ) - x(Θ1)||. The paper
    says it can get stuck in local optima; the learned network is meant to do better.
    It is NOT a function of position alone (two points at the same place can have
    different targets), so the non-intersection argument does not strictly apply.
    Use it as a sanity check of the pipeline and as a baseline for the network."""

    def forward(self, x, theta_t, theta_1, dt, x_target=None):
        return (x_target - x) / dt.clamp_min(DT_MIN)[:, None, None]


class SmoothTargetField(torch.nn.Module):
    """Training-free field that IS a smooth function of position (our own addition,
    not in the paper). The per-point target velocities v_j = (x_j(Θ1) - x_j) / Δt
    are spread over space with a Gaussian kernel of width sigma (metres):

        f(y) = Σ_j K(y, x_j) v_j / Σ_j K(y, x_j),   K = exp(-||y - x_j||² / 2σ²)

    Because f depends only on the position y, points that touch get (almost) the
    same velocity and cannot pass through each other (Supp. A, CLOAF.mp4 remark).
    Small sigma -> follows the target closely; large sigma -> safer but blurrier."""

    def __init__(self, sigma: float = 0.03):
        super().__init__()
        self.sigma = sigma

    def forward(self, x, theta_t, theta_1, dt, x_target=None):
        v = (x_target - x) / dt.clamp_min(DT_MIN)[:, None, None]       # (B,S,3)
        d2 = torch.cdist(x, x) ** 2                                      # (B,S,S)
        K = torch.softmax(-d2 / (2 * self.sigma ** 2), dim=-1)          # normalised kernel
        return K @ v


class TargetField(torch.nn.Module):
    """f(x) = F (x_T - x) / (||x_T - x|| + eps) in the source region, blended to 0 outside.
    r_in = 10 mm, r_out = 30 mm, F = 1e-3, eps = 1e-6 (Supp. C).
    Optional axis-aligned box obstacle -> zero field inside the box (Fig. 4)."""

    def __init__(self, body, region_vert_mask, target_point, F=1e-3, eps=1e-6,
                 r_in=0.010, r_out=0.030, box=None):
        super().__init__()
        self.body, self.mask, self.xT = body, region_vert_mask, target_point
        self.F, self.eps, self.r_in, self.r_out, self.box = F, eps, r_in, r_out, box

    def forward(self, x, theta_t, theta_1, dt, x_target=None):
        verts = self.body.vertices(theta_t.to(self.body.dtype)).float()   # (B,V,3)
        region = verts[:, self.mask]                                      # (B,R,3)
        r = torch.cdist(x, region).min(-1).values                         # (B,S) distance to region
        b = bezier_blend(r, self.r_in, self.r_out)
        d = self.xT.to(x) - x
        f = self.F * d / (d.norm(dim=-1, keepdim=True) + self.eps)
        f = f * (1 - b)[..., None]                                        # Eq. 11
        if self.box is not None:                                          # obstacle
            lo, hi = (t.to(x) for t in self.box)
            dist_out = (torch.maximum(lo - x, x - hi)).clamp_min(0).norm(dim=-1)
            f = f * bezier_blend(dist_out, 0.0, self.r_out - self.r_in)[..., None]
        return f


# ====================================================================== #
#                      Sequence (video) correction                         #
# ====================================================================== #
@torch.no_grad()
def correct_sequence(body, field, poses, betas=None, global_orient=None,
                     mode="all", init_strategy="jitter", S=1000, progress=False,
                     return_eps=None, return_decay=0.7, **solve_kw):
    """
    poses : (N,69) estimated poses (may contain collisions)
    mode  : "all"            -> paper: every frame, Θ0 = previous corrected, Θ1 = estimate
            "colliding_only" -> extension: collision-free frames are kept unchanged
    return_eps (m, colliding_only only): "return phase" (our extension). After a corrected
            stretch, the following collision-free frames are not reset to the input at once:
            they keep the last correction offset Δ = Θ_out - Θ_in, shrunk by return_decay every
            frame (Θ = Θ_in + decay^j Δ), until its effect is below return_eps (mean vertex
            distance) or the next collision. A pose of the return phase that collides is
            corrected with CLOAF as usual. None = off (jump back in one frame).
    Returns: (N,69) corrected poses, (N,) collision counts before correction
    """
    N = poses.shape[0]
    sel = lambda x, i: None if x is None else x[i:i + 1]
    verts = body.vertices(poses, betas, global_orient)
    counts = body.count_collisions(verts)
    out = poses.clone().to(body.dtype)
    last_free = None          # most recent output pose known to be collision-free
    skipped = []
    stats = {"previous frame": 0, "last free pose": 0, "jitter": 0, "skipped": 0, "failed": 0,
             "return phase": 0}
    returning = False          # inside a return phase (see return_eps)
    offset, j_ret = None, 0    # last correction offset in Θ, frames since the last correction

    def collides(theta, k):
        return body.count_collisions(body.vertices(theta, sel(betas, k), sel(global_orient, k)))[0] > body.collision_min

    def start_pose(candidate, k):
        """Collision-free Θ0 for frame k, cheapest option first:
        previous output -> last collision-free output -> a short jitter search."""
        if candidate is not None and not collides(candidate, k):
            stats["previous frame"] += 1
            return candidate
        if last_free is not None and not collides(last_free, k):
            stats["last free pose"] += 1
            return last_free
        try:
            th = find_initial_pose(body, candidate if candidate is not None else poses[k:k + 1],
                                   sel(betas, k), sel(global_orient, k), strategy=init_strategy,
                                   max_iter=20)
            stats["jitter"] += 1
            return th
        except RuntimeError:
            return None

    import time
    t_start = time.time()
    for k in range(N):
        if progress and k % 10 == 0:
            print(f"  frame {k}/{N} ({time.time() - t_start:.0f} s)", flush=True)
        if mode == "colliding_only" and counts[k] <= body.collision_min:
            if returning:
                j_ret += 1
                cand = poses[k:k + 1].to(body.dtype) + return_decay ** j_ret * offset
                gap = (body.vertices(cand, sel(betas, k), sel(global_orient, k))
                       - verts[k:k + 1]).norm(dim=-1).mean().item()
                if gap <= return_eps:
                    returning = False                 # close enough: back to the input
                elif not collides(cand, k):
                    out[k] = cand[0]                  # decayed correction, collision-free
                    last_free = cand.clone()
                    stats["return phase"] += 1
                    continue
                # else: the decayed pose collides -> correct this frame with CLOAF below
            if not returning:
                last_free = out[k:k + 1].clone()      # keep the original frame
                continue
        if k == 0 and counts[0] <= body.collision_min:
            last_free = out[0:1].clone()              # nothing to correct, start from here
            continue
        # Θ0 = previous output; re-check it with this frame's betas/orient (the shape may change!)
        theta_0 = start_pose(out[k - 1:k] if k > 0 else poses[0:1], k)
        if theta_0 is None:                           # no collision-free start: leave frame as is
            skipped.append(k)
            stats["skipped"] += 1
            continue
        reason = None
        try:
            traj, _ = cloaf_solve(body, field, theta_0, poses[k:k + 1], S,
                                  sel(betas, k), sel(global_orient, k), **solve_kw)
            if not torch.isfinite(traj[-1]).all():
                reason = "non-finite pose"
        except (RuntimeError, torch.linalg.LinAlgError) as e:   # numerical failure in this frame
            reason = type(e).__name__
        if reason is not None:
            stats["failed"] += 1
            if stats["failed"] <= 3:
                print(f"  frame {k}: integration failed ({reason}), frame left unchanged", flush=True)
            continue
        out[k] = traj[-1, 0]
        if return_eps is not None and mode == "colliding_only":
            returning, j_ret = True, 0                # return to the input gradually afterwards
            offset = out[k:k + 1] - poses[k:k + 1].to(body.dtype)
        if not collides(out[k:k + 1], k):
            last_free = out[k:k + 1].clone()
    if progress:
        print(f"  start pose used: {stats}", flush=True)
    if skipped:
        print(f"warning: no collision-free start pose for {len(skipped)} frame(s), "
              f"left unchanged: {skipped[:10]}{' ...' if len(skipped) > 10 else ''}")
    return out, counts


# ====================================================================== #
#                  Initial pose: Jitter (Sec. 4.5)                         #
# ====================================================================== #
@torch.no_grad()
def find_initial_pose(body, theta, betas=None, global_orient=None, strategy="jitter",
                      sigma0=0.05, max_iter=20, keyposes=None):
    """Find a collision-free Θ0.
    jitter  : add noise to the joints driving the colliding vertices, increase σ over time.
    keypose : nearest of the AMASS K-means key poses (K=128) -> keyposes (K,69)."""
    if strategy == "keypose":
        assert keyposes is not None
        d = (keyposes - theta).norm(dim=-1)   # TODO: the paper uses keypoint distance
        return keyposes[d.argmin()][None].to(theta)

    W = body.smpl.lbs_weights                           # (V,24)
    th = theta.clone().to(body.dtype)
    sigma = sigma0
    for it in range(max_iter):
        v = body.vertices(th, betas, global_orient)[0]
        m = body.inside(v)
        if m.sum() == 0:
            return th
        joints = W[m].sum(0).topk(4).indices            # joints responsible for the collision
        joints = joints[joints > 0] - 1                 # convert to body_pose index (no pelvis)
        cand = theta.clone().to(body.dtype)
        for j in joints.tolist():
            cand[0, 3 * j:3 * j + 3] += sigma * torch.randn(3, device=th.device, dtype=th.dtype)
        th = cand
        if it % 5 == 4:
            sigma *= 1.5
    raise RuntimeError("Jitter could not find a collision-free pose")
