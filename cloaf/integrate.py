"""
CLOAF entegrasyonu (Denk. 6):   dΘ/dt = J^† f(X(Θ), t | Θ0, Θ1),   Θ(t0) = Θ0

- Öğrenilmiş alan  : MotionField (model.py)
- Basit alan (4.4) : TargetField + Bézier blending (Denk. 10-11, Supp. C.14)
- Dizi düzeltme    : "successive frames" (Sec. 4.2) ve uzantı olarak
                     "colliding_only" (WS proje önerisi)
"""
from __future__ import annotations

import torch
from torchdiffeq import odeint

from .body import SMPLBody, pinv_solve
from .collision import count_collisions


# ====================================================================== #
#                           ODE sağ tarafı                                #
# ====================================================================== #
class CLOAFODE(torch.nn.Module):
    """Θ uzayında hız: J^† f. torchdiffeq.odeint ile çözülür."""

    def __init__(self, body: SMPLBody, field, theta_1, sample, betas=None,
                 global_orient=None, t1: float = 1.0, jac_method="fd"):
        super().__init__()
        self.body, self.field, self.sample = body, field, sample
        self.theta_1, self.betas, self.go, self.t1 = theta_1, betas, global_orient, t1
        self.jac_method = jac_method
        self.nfe = 0
        with torch.no_grad():
            self.x_1 = body.points(theta_1, sample, betas, global_orient)

    def forward(self, t, theta):
        self.nfe += 1
        x = self.body.points(theta, self.sample, self.betas, self.go)
        J = self.body.jacobian(theta, self.sample, self.betas, self.go, method=self.jac_method)
        dt = (self.t1 - t).expand(theta.shape[0]).to(theta)
        f = self.field(x.float(), theta.float(), self.theta_1.float(), dt.float(),
                       x_target=self.x_1.float())                     # (B,S,3)
        return pinv_solve(J, f.reshape(f.shape[0], -1))              # (B,69)


@torch.no_grad()
def cloaf_solve(body, field, theta_0, theta_1, S=1000, betas=None, global_orient=None,
                method="dopri5", rtol=1e-4, atol=1e-5, n_out=2, seed=0, **kw):
    """Θ0 (çakışmasız) -> Θ1 hedefe entegre et. Dönüş: (n_out, B, 69) yörünge."""
    g = torch.Generator(device=body.device).manual_seed(seed)
    sample = body.sample_surface(S, generator=g)
    ode = CLOAFODE(body, field, theta_1.to(body.dtype), sample, betas, global_orient, **kw)
    ts = torch.linspace(0, 1, n_out, device=body.device, dtype=body.dtype)
    traj = odeint(ode, theta_0.to(body.dtype), ts, method=method, rtol=rtol, atol=atol)
    return traj, ode.nfe


# ====================================================================== #
#                 Basit (elle tanımlı) alan — Sec. 4.4 / Supp. C           #
# ====================================================================== #
def bezier_blend(r, r_in, r_out, alphas=(0, 0, 0, 1, 1)):
    """Denk. 10: b(r) = Σ α_p B_p^4((r-r_in)/(r_out-r_in))."""
    from math import comb
    u = ((r - r_in) / (r_out - r_in)).clamp(0, 1)
    return sum(a * comb(4, p) * u ** p * (1 - u) ** (4 - p) for p, a in enumerate(alphas))


class TargetField(torch.nn.Module):
    """f(x) = F (x_T - x) / (||x_T - x|| + eps)  kaynak bölgede, dışarıda blend ile 0.
    r_in = 10 mm, r_out = 30 mm, F = 1e-3, eps = 1e-6 (Supp. C).
    Opsiyonel: eksen hizalı kutu engel -> kutu içinde alan 0 (Fig. 4)."""

    def __init__(self, body, region_vert_mask, target_point, F=1e-3, eps=1e-6,
                 r_in=0.010, r_out=0.030, box=None):
        super().__init__()
        self.body, self.mask, self.xT = body, region_vert_mask, target_point
        self.F, self.eps, self.r_in, self.r_out, self.box = F, eps, r_in, r_out, box

    def forward(self, x, theta_t, theta_1, dt, x_target=None):
        verts = self.body.vertices(theta_t.to(self.body.dtype)).float()   # (B,V,3)
        region = verts[:, self.mask]                                      # (B,R,3)
        r = torch.cdist(x, region).min(-1).values                         # (B,S)
        b = bezier_blend(r, self.r_in, self.r_out)
        d = self.xT.to(x) - x
        f = self.F * d / (d.norm(dim=-1, keepdim=True) + self.eps)
        f = f * (1 - b)[..., None]                                        # Denk. 11
        if self.box is not None:                                          # engel
            lo, hi = (t.to(x) for t in self.box)
            dist_out = (torch.maximum(lo - x, x - hi)).clamp_min(0).norm(dim=-1)
            f = f * bezier_blend(dist_out, 0.0, self.r_out - self.r_in)[..., None]
        return f


# ====================================================================== #
#                     Dizi (video) düzeltme                                #
# ====================================================================== #
@torch.no_grad()
def correct_sequence(body, field, poses, betas=None, global_orient=None,
                     mode="all", init_strategy="jitter", S=1000, **solve_kw):
    """
    poses : (N,69) tahmin edilen (çakışmalı olabilir) pozlar
    mode  : "all"            -> makale: her frame Θ0=önceki düzeltilmiş, Θ1=tahmin
            "colliding_only" -> uzantı: çakışmasız frame'ler olduğu gibi kalır
    Dönüş : (N,69) düzeltilmiş pozlar, (N,) düzeltme öncesi çakışma sayıları
    """
    N = poses.shape[0]
    sel = lambda x, i: None if x is None else x[i:i + 1]
    verts = body.vertices(poses, betas, global_orient)
    counts = count_collisions(verts, body.faces)
    out = poses.clone().to(body.dtype)

    # --- ilk frame ---
    if counts[0] > 0:
        out[0] = find_initial_pose(body, poses[0:1], sel(betas, 0), sel(global_orient, 0),
                                   strategy=init_strategy)[0]
        out[0] = cloaf_solve(body, field, out[0:1], poses[0:1], S, sel(betas, 0),
                             sel(global_orient, 0), **solve_kw)[0][-1, 0]

    for k in range(1, N):
        if mode == "colliding_only" and counts[k] == 0:
            continue                                    # orijinal frame korunur
        theta_0 = out[k - 1:k]
        # Θ0'ı hedef frame'in betas/orient'i ile kontrol et (şekil değişebilir!)
        v0 = body.vertices(theta_0, sel(betas, k), sel(global_orient, k))
        if count_collisions(v0, body.faces)[0] > 0:
            theta_0 = find_initial_pose(body, theta_0, sel(betas, k), sel(global_orient, k),
                                        strategy=init_strategy)
        traj, _ = cloaf_solve(body, field, theta_0, poses[k:k + 1], S,
                              sel(betas, k), sel(global_orient, k), **solve_kw)
        out[k] = traj[-1, 0]
    return out, counts


# ====================================================================== #
#                 Başlangıç pozu: Jitter (Sec. 4.5)                        #
# ====================================================================== #
@torch.no_grad()
def find_initial_pose(body, theta, betas=None, global_orient=None, strategy="jitter",
                      sigma0=0.05, max_iter=200, keyposes=None):
    """Çakışmasız bir Θ0 bul.
    jitter  : çakışan vertexlerin bağlı olduğu eklemlere gürültü ekle, σ'yı artır.
    keypose : AMASS K-means anahtar pozlarından en yakını (K=128) -> keyposes (K,69)."""
    if strategy == "keypose":
        assert keyposes is not None
        d = (keyposes - theta).norm(dim=-1)   # TODO: makale keypoint mesafesi kullanıyor
        return keyposes[d.argmin()][None].to(theta)

    from .collision import inside_vertices
    W = body.smpl.lbs_weights                           # (V,24)
    th = theta.clone().to(body.dtype)
    sigma = sigma0
    for it in range(max_iter):
        v = body.vertices(th, betas, global_orient)[0]
        m = inside_vertices(v, body.faces)
        if m.sum() == 0:
            return th
        joints = W[m].sum(0).topk(4).indices            # sorumlu eklemler
        joints = joints[joints > 0] - 1                 # body_pose indeksine çevir (pelvis hariç)
        cand = theta.clone().to(body.dtype)
        for j in joints.tolist():
            cand[0, 3 * j:3 * j + 3] += sigma * torch.randn(3, device=th.device, dtype=th.dtype)
        th = cand
        if it % 20 == 19:
            sigma *= 1.5
    raise RuntimeError("Jitter çakışmasız poz bulamadı")
