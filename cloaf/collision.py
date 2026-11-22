"""
Self-intersection (öz-kesişim) tespiti: Generalized Winding Number (GWN).

Kapalı bir yüzeyde bir vertex için GWN ≈ 0.5'tir (kendi yüzeyinin üzerinde).
Vertex vücudun başka bir parçasının İÇİNDEYSE GWN ≈ 1.5 olur.
=> threshold 1.0 ile "vücudun içindeki vertex" sayılır.

Makale metriği (Sec. 4.1): Col.Rate@C = vücudun içinde > C vertex bulunan
örneklerin (frame'lerin) yüzdesi. Makale tespit yöntemini belirtmiyor!
=> ÖNCE Table 1'deki HMR2.0 değerlerini (39.2 / 21.6) bu tespit edici ile
   yeniden üretmeye çalış. Uyuşmazsa threshold / hariç bölgeleri tartış.

NOT: SMPL mesh'inin watertight olduğunu trimesh ile doğrula (mesh.is_watertight).
"""
import math
import torch


def winding_numbers(verts, faces, queries=None, chunk: int = 512):
    """verts (V,3), faces (F,3), queries (N,3) -> (N,) GWN. GPU'da çalıştır."""
    if queries is None:
        queries = verts
    tri = verts[faces]                                       # (F,3,3)
    out = []
    for q in queries.split(chunk):
        a, b, c = (tri[None, :, k] - q[:, None] for k in range(3))   # (n,F,3)
        la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
        det = (a * torch.linalg.cross(b, c)).sum(-1)
        den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
        omega = 2 * torch.atan2(det, den)                    # işaretli katı açı
        out.append(omega.sum(-1) / (4 * math.pi))
    return torch.cat(out)


@torch.no_grad()
def inside_vertices(verts, faces, thresh: float = 1.0, ignore_mask=None):
    """Vücudun içinde kalan vertex maskesi (V,). ignore_mask: sayılmayacak vertexler."""
    w = winding_numbers(verts.float(), faces)
    m = w > thresh
    if ignore_mask is not None:
        m &= ~ignore_mask
    return m


@torch.no_grad()
def count_collisions(verts_batch, faces, **kw):
    """(B,V,3) -> (B,) vücut içindeki vertex sayısı."""
    return torch.stack([inside_vertices(v, faces, **kw).sum() for v in verts_batch])


def collision_rate(counts, C: int = 0):
    """Col.Rate@C (%): count > C olan frame oranı."""
    counts = torch.as_tensor(counts)
    return 100.0 * (counts > C).float().mean().item()
