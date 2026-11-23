"""
SMPL wrapper: surface point sampling, X(Θ) and the Jacobian J = dX/dΘ.

Mapping to the paper:
  - Θ  : 69-dim body_pose (23 joints x axis-angle). Supp. Fig. B.8 plots the
         components i ∈ [0, 69) -> global_orient is NOT part of Θ.
  - S  : number of points sampled on the surface (default 1000, Sec. 4.5 / Fig. 5).
  - J  : R^{3S x d} Jacobian, computed in double precision (float64, Supp. B).
  - Crotch and armpit regions are excluded from sampling (Sec. 3).
"""
from __future__ import annotations

import torch
import smplx

from .collision import PartCollisionTest, inside_vertices

POSE_DIM = 69  # body_pose

# SMPL joint indices
PELVIS, L_HIP, R_HIP = 0, 1, 2
L_COLLAR, R_COLLAR, L_SHOULDER, R_SHOULDER = 13, 14, 16, 17
SPINE3 = 9


class SMPLBody(torch.nn.Module):
    def __init__(self, model_path: str, gender: str = "neutral",
                 device="cuda", dtype=torch.float64, exclude_radius: float = 0.07,
                 collision_mode: str = "parts", ignore_rest_collisions: bool = True,
                 ignore_excluded_regions: bool = True, collision_min: int = 0):
        super().__init__()
        self.device, self.dtype = torch.device(device), dtype
        self.smpl = smplx.SMPL(model_path, gender=gender).to(self.device, dtype)
        for p in self.smpl.parameters():
            p.requires_grad_(False)
        self.faces = torch.as_tensor(self.smpl.faces.astype("int64"), device=self.device)

        # Face areas in the T-pose and the mask of excluded regions
        with torch.no_grad():
            out = self.smpl(betas=torch.zeros(1, 10, device=self.device, dtype=dtype))
            v, j = out.vertices[0], out.joints[0]
        self.register_buffer("excluded_verts", self._exclusion_mask(v, j, exclude_radius))
        tri = v[self.faces]
        area = 0.5 * torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).norm(dim=-1)
        bad_face = self.excluded_verts[self.faces].any(-1)
        area[bad_face] = 0.0
        self.register_buffer("face_prob", area / area.sum())

        # Collision test (see collision.py): "parts" ignores contact between adjacent
        # body parts; "full" counts every vertex inside the body.
        self.collision_mode = collision_mode
        # A pose counts as colliding only with MORE than collision_min colliding vertices.
        # 0 = any penetration (paper's Col.Rate@0). A small value (e.g. 20) tolerates the
        # light hand-body contact that clean mocap has in ~30 % of frames.
        self.collision_min = collision_min
        self._part_test = (PartCollisionTest(self.smpl.lbs_weights, self.faces)
                           if collision_mode == "parts" else None)
        # Ignored when counting: vertices the test already flags in the rest pose
        # (a property of the template, not of the pose) and, optionally, the
        # crotch/armpit regions that the paper also excludes from sampling.
        rest = self._inside_raw(v)
        self.register_buffer("rest_inside", rest)
        ignore = torch.zeros_like(rest)
        if ignore_rest_collisions:
            ignore |= rest
        if ignore_excluded_regions:
            ignore |= self.excluded_verts
        self.register_buffer("collision_ignore", ignore)

    # ------------------------------------------------------------------ #
    def _inside_raw(self, verts):
        if self._part_test is not None:
            return self._part_test(verts)
        return inside_vertices(verts, self.faces)

    def inside(self, verts):
        """(V,3) -> (V,) mask of colliding vertices (ignored vertices removed)."""
        return self._inside_raw(verts) & ~self.collision_ignore

    @torch.no_grad()
    def count_collisions(self, verts_batch):
        """(B,V,3) -> (B,) number of colliding vertices (ignored vertices removed)."""
        return torch.stack([self.inside(v).sum() for v in verts_batch])

    # ------------------------------------------------------------------ #
    @staticmethod
    def _exclusion_mask(v, j, r):
        """Vertices near the crotch and the armpits.
        TODO: check visually (colour the mask in aitviewer) and tune r.
        The paper does not define these regions; it only cites [26] (TUCH)."""
        down = torch.tensor([0.0, 0.06, 0.0], dtype=v.dtype, device=v.device)
        crotch = 0.5 * (j[L_HIP] + j[R_HIP]) - down
        l_arm = j[L_SHOULDER] - down
        r_arm = j[R_SHOULDER] - down
        centers = torch.stack([crotch, l_arm, r_arm])
        return (torch.cdist(v, centers) < r).any(-1)

    # ------------------------------------------------------------------ #
    def sample_surface(self, S: int, generator=None):
        """Area-weighted uniform surface sampling -> (face_idx (S,), bary (S,3)).
        Pass the result as `sample` to points()/jacobian(). Use sample=None
        to take all 6890 vertices instead (S = 6890, Supp. B.7)."""
        fidx = torch.multinomial(self.face_prob, S, replacement=True, generator=generator)
        u = torch.rand(S, 2, device=self.device, dtype=self.dtype, generator=generator)
        su = u[:, 0].sqrt()
        bary = torch.stack([1 - su, su * (1 - u[:, 1]), su * u[:, 1]], -1)
        return fidx, bary

    # ------------------------------------------------------------------ #
    def vertices(self, body_pose, betas=None, global_orient=None):
        B = body_pose.shape[0]
        z = lambda n: torch.zeros(B, n, device=self.device, dtype=self.dtype)
        out = self.smpl(body_pose=body_pose.to(self.dtype),
                        betas=z(10) if betas is None else betas.to(self.dtype),
                        global_orient=z(3) if global_orient is None else global_orient.to(self.dtype))
        return out.vertices  # (B, 6890, 3)

    def points(self, body_pose, sample, betas=None, global_orient=None):
        """X(Θ): (B, S, 3). sample=None -> all vertices."""
        v = self.vertices(body_pose, betas, global_orient)
        if sample is None:
            return v
        fidx, bary = sample
        tri = v[:, self.faces[fidx]]                 # (B, S, 3, 3)
        return (tri * bary[None, :, :, None]).sum(2)

    # ------------------------------------------------------------------ #
    def jacobian(self, body_pose, sample, betas=None, global_orient=None,
                 method: str = "fd", eps: float = None):
        """J = dX/dΘ, shape (B, 3S, 69), float64.

        method="fd"     : central finite differences (very accurate in float64, always works)
        method="jacfwd" : torch.func forward-mode AD (may be faster; compare it with fd!)
        """
        th = body_pose.to(self.dtype)
        B, d = th.shape
        if eps is None:
            eps = 1e-6 if self.dtype == torch.float64 else 1e-3
        if method == "jacfwd":
            # per-sample forward-mode AD, vectorised over the batch: cost O(B * d)
            from torch.func import jacfwd, vmap
            bt = torch.zeros(B, 10, device=self.device, dtype=self.dtype) if betas is None else betas.to(self.dtype)
            go = (torch.zeros(B, 3, device=self.device, dtype=self.dtype) if global_orient is None
                  else global_orient.to(self.dtype))

            def f(t, b, g):
                return self.points(t[None], sample, b[None], g[None])[0].reshape(-1)
            return vmap(jacfwd(f, argnums=0))(th, bt, go)   # (B, 3S, d)

        # finite differences: B*(2d) SMPL evaluations in a single batch
        E = torch.eye(d, device=self.device, dtype=self.dtype) * eps
        tp = (th[:, None] + E[None]).reshape(-1, d)
        tm = (th[:, None] - E[None]).reshape(-1, d)
        rep = lambda x: None if x is None else x.repeat_interleave(d, 0)
        xp = self.points(tp, sample, rep(betas), rep(global_orient)).reshape(B, d, -1)
        xm = self.points(tm, sample, rep(betas), rep(global_orient)).reshape(B, d, -1)
        return ((xp - xm) / (2 * eps)).transpose(1, 2)   # (B, 3S, d)


def contact_constrained_solve(J, f, x, far_pairs, dist: float = 0.05, mu: float = 1e3,
                              iters: int = 4, max_pairs: int = 400, damping: float = 1e-10):
    """Least squares dΘ ≈ J^+ f, but close pairs of points on non-adjacent body parts
    may not approach each other (our extension, not in the paper).

    For every pair (i, j) closer than `dist` with n = (x_i - x_j)/|x_i - x_j|, the rate of
    change of their distance is a·dΘ with a = n^T (J_i - J_j). Pairs with a·dΘ < 0 (approaching)
    become active and a·dΘ = 0 is enforced with a large penalty; repeated `iters` times
    (a simple active-set scheme). Sliding along the contact stays allowed.
    J (B,3S,d), f (B,3S), x (B,S,3), far_pairs (S,S) bool. Returns dΘ (B,d) and the number
    of active constraints per sample."""
    J = J.double()
    B, _, d = J.shape
    S = x.shape[1]
    Jp = J.reshape(B, S, 3, d)
    I = torch.eye(d, dtype=J.dtype, device=J.device)
    out, n_active = [], []
    for b in range(B):
        Jb, fb, xb = J[b], f[b].double(), x[b].double()
        JtJ, rhs = Jb.T @ Jb, Jb.T @ fb
        scale = JtJ.diagonal().mean()
        lam = damping * scale + 1e-12
        D = torch.cdist(xb, xb)
        D = D.masked_fill(~far_pairs, float("inf"))
        D = torch.triu(D, diagonal=1) + torch.tril(torch.full_like(D, float("inf")))
        ii, jj = (D < dist).nonzero(as_tuple=True)
        if len(ii) > max_pairs:
            keep = D[ii, jj].argsort()[:max_pairs]
            ii, jj = ii[keep], jj[keep]
        dth = torch.linalg.solve(JtJ + lam * I, rhs)
        if len(ii) == 0:
            out.append(dth)
            n_active.append(0)
            continue
        n = xb[ii] - xb[jj]
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        A = torch.einsum("pc,pcd->pd", n, Jp[b, ii] - Jp[b, jj])            # (P,d)
        mu_eff = mu * scale / A.pow(2).sum(-1).mean().clamp_min(1e-12)
        active = torch.zeros(len(ii), dtype=torch.bool, device=J.device)
        for _ in range(iters):
            viol = (A @ dth) < 0
            if not (viol & ~active).any():
                break
            active |= viol
            Aa = A[active]
            dth = torch.linalg.solve(JtJ + mu_eff * Aa.T @ Aa + lam * I, rhs)
        out.append(dth)
        n_active.append(int(active.sum()))
    return torch.stack(out), n_active


def pinv_solve(J, f, damping: float = 1e-10, weights=None):
    """dΘ = J^† f  (Eq. 5), J: (B,3S,d), f: (B,3S).
    Least-squares solution via the normal equations with a tiny damping term.
    weights (B,S) or None: weighted least squares dΘ = (J^T W J)^-1 J^T W f,
    W = one weight per point, repeated over its x, y, z rows (None = paper)."""
    J = J.double()
    Jt = J.transpose(1, 2)
    if weights is not None:
        Jt = Jt * weights.double().repeat_interleave(3, dim=-1)[:, None, :]
    JtJ = Jt @ J
    lam = damping * JtJ.diagonal(dim1=1, dim2=2).mean(-1, keepdim=True)[..., None] + 1e-12
    I = torch.eye(J.shape[-1], dtype=J.dtype, device=J.device)
    rhs = Jt @ f.double()[..., None]
    return torch.linalg.solve(JtJ + lam * I, rhs)[..., 0]
