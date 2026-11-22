"""
SMPL wrapper: yüzey nokta örnekleme, X(Θ) ve Jacobian J = dX/dΘ.

Makaledeki karşılıklar:
  - Θ  : 69 boyutlu body_pose (23 eklem x axis-angle). Supp. Fig. B.8'de
         bileşenler i ∈ [0, 69) olarak çizilmiş -> global_orient Θ'ya dahil değil.
  - S  : yüzeyden örneklenen nokta sayısı (varsayılan 1000, Sec. 4.5 / Fig. 5).
  - J  : R^{3S x d} Jacobian, çift hassasiyette (float64) hesaplanır (Supp. B).
  - Kasık (crotch) ve koltuk altı (armpit) bölgeleri örneklemeden çıkarılır (Sec. 3).
"""
from __future__ import annotations

import torch
import smplx

POSE_DIM = 69  # body_pose

# SMPL eklem indeksleri
PELVIS, L_HIP, R_HIP = 0, 1, 2
L_COLLAR, R_COLLAR, L_SHOULDER, R_SHOULDER = 13, 14, 16, 17
SPINE3 = 9


class SMPLBody(torch.nn.Module):
    def __init__(self, model_path: str, gender: str = "neutral",
                 device="cuda", dtype=torch.float64, exclude_radius: float = 0.07):
        super().__init__()
        self.device, self.dtype = torch.device(device), dtype
        self.smpl = smplx.SMPL(model_path, gender=gender).to(self.device, dtype)
        for p in self.smpl.parameters():
            p.requires_grad_(False)
        self.faces = torch.as_tensor(self.smpl.faces.astype("int64"), device=self.device)

        # T-pozda yüzey alanları ve hariç tutulacak bölge maskesi
        with torch.no_grad():
            out = self.smpl(betas=torch.zeros(1, 10, device=self.device, dtype=dtype))
            v, j = out.vertices[0], out.joints[0]
        self.register_buffer("excluded_verts", self._exclusion_mask(v, j, exclude_radius))
        tri = v[self.faces]
        area = 0.5 * torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).norm(dim=-1)
        bad_face = self.excluded_verts[self.faces].any(-1)
        area[bad_face] = 0.0
        self.register_buffer("face_prob", area / area.sum())

    # ------------------------------------------------------------------ #
    @staticmethod
    def _exclusion_mask(v, j, r):
        """Kasık ve koltuk altı yakınındaki vertexler.
        TODO: Görsel olarak kontrol et (aitviewer'da maskeyi renklendir) ve r'yi ayarla.
        Makale bölgelerin nasıl tanımlandığını vermiyor ([26] TUCH'a atıf yapıyor)."""
        crotch = 0.5 * (j[L_HIP] + j[R_HIP])
        crotch = crotch - torch.tensor([0.0, 0.06, 0.0], dtype=v.dtype, device=v.device)
        l_arm = j[L_SHOULDER] - torch.tensor([0.0, 0.06, 0.0], dtype=v.dtype, device=v.device)
        r_arm = j[R_SHOULDER] - torch.tensor([0.0, 0.06, 0.0], dtype=v.dtype, device=v.device)
        centers = torch.stack([crotch, l_arm, r_arm])
        return (torch.cdist(v, centers) < r).any(-1)

    # ------------------------------------------------------------------ #
    def sample_surface(self, S: int, generator=None):
        """Alan ağırlıklı uniform yüzey örneklemesi -> (face_idx (S,), bary (S,3))."""
        fidx = torch.multinomial(self.face_prob, S, replacement=True, generator=generator)
        u = torch.rand(S, 2, device=self.device, dtype=self.dtype, generator=generator)
        su = u[:, 0].sqrt()
        bary = torch.stack([1 - su, su * (1 - u[:, 1]), su * u[:, 1]], -1)
        return fidx, bary

    def all_vertices_sample(self):
        """S = 6890: tüm vertexler (Supp. B.7 deneyi için)."""
        return None

    # ------------------------------------------------------------------ #
    def vertices(self, body_pose, betas=None, global_orient=None):
        B = body_pose.shape[0]
        z = lambda n: torch.zeros(B, n, device=self.device, dtype=self.dtype)
        out = self.smpl(body_pose=body_pose.to(self.dtype),
                        betas=z(10) if betas is None else betas.to(self.dtype),
                        global_orient=z(3) if global_orient is None else global_orient.to(self.dtype))
        return out.vertices  # (B, 6890, 3)

    def points(self, body_pose, sample, betas=None, global_orient=None):
        """X(Θ): (B, S, 3). sample=None -> tüm vertexler."""
        v = self.vertices(body_pose, betas, global_orient)
        if sample is None:
            return v
        fidx, bary = sample
        tri = v[:, self.faces[fidx]]                 # (B, S, 3, 3)
        return (tri * bary[None, :, :, None]).sum(2)

    # ------------------------------------------------------------------ #
    def jacobian(self, body_pose, sample, betas=None, global_orient=None,
                 method: str = "fd", eps: float = 1e-6):
        """J = dX/dΘ, şekil (B, 3S, 69), float64.

        method="fd"     : merkezi sonlu farklar (float64'te çok doğru, her zaman çalışır)
        method="jacfwd" : torch.func ileri-mod AD (daha hızlı olabilir; fd ile karşılaştır!)
        """
        th = body_pose.to(self.dtype)
        B, d = th.shape
        if method == "jacfwd":
            from torch.func import jacfwd

            def G(t):  # örnekler bağımsız -> toplamın Jacobian'ı blok-diyagonal bilgiyi verir
                return self.points(t, sample, betas, global_orient).reshape(B, -1).sum(0)
            Jfull = jacfwd(G)(th)                    # (3S, B, d)
            return Jfull.permute(1, 0, 2).contiguous()

        # sonlu farklar: B*(2d) SMPL değerlendirmesi tek batch'te
        E = torch.eye(d, device=self.device, dtype=self.dtype) * eps
        tp = (th[:, None] + E[None]).reshape(-1, d)
        tm = (th[:, None] - E[None]).reshape(-1, d)
        rep = lambda x: None if x is None else x.repeat_interleave(d, 0)
        xp = self.points(tp, sample, rep(betas), rep(global_orient)).reshape(B, d, -1)
        xm = self.points(tm, sample, rep(betas), rep(global_orient)).reshape(B, d, -1)
        return ((xp - xm) / (2 * eps)).transpose(1, 2)   # (B, 3S, d)


def pinv_solve(J, f, damping: float = 1e-10):
    """dΘ = J^† f  (Denk. 5), J: (B,3S,d), f: (B,3S). Normal denklemler + küçük sönüm."""
    J = J.double()
    JtJ = J.transpose(1, 2) @ J
    lam = damping * JtJ.diagonal(dim1=1, dim2=2).mean(-1, keepdim=True)[..., None]
    I = torch.eye(J.shape[-1], dtype=J.dtype, device=J.device)
    rhs = J.transpose(1, 2) @ f.double()[..., None]
    return torch.linalg.solve(JtJ + lam * I, rhs)[..., 0]
