"""
Self-intersection detection with the Generalized Winding Number (GWN).

For a vertex on a closed surface the GWN is ≈ 0.5 (it lies on its own surface).
If the vertex is INSIDE another body part, the GWN is ≈ 1.5.
=> with threshold 1.0 we count "vertices inside the body".

Paper metric (Sec. 4.1): Col.Rate@C = percentage of samples (frames) with
more than C vertices inside the body. The paper does NOT describe how this
is detected!
=> FIRST try to reproduce the HMR2.0 numbers of Table 1 (39.2 / 21.6) with
   this detector. If they do not match, discuss the threshold / excluded regions.

NOTE: the SMPL mesh is watertight (checked with tests/check_smpl.py). Vertices
that already look "inside" in the rest pose are not real collisions; SMPLBody
excludes them by default (SMPLBody.collision_ignore).
"""
import math
import torch


def winding_numbers(verts, faces, queries=None, chunk: int = 256, dtype=torch.float64):
    """verts (V,3), faces (F,3), queries (N,3) -> (N,) GWN. Run on GPU for speed.
    float64 by default: near-touching surfaces give wrong values in float32."""
    verts = verts.to(dtype)
    queries = verts if queries is None else queries.to(dtype)
    tri = verts[faces]                                       # (F,3,3)
    out = []
    for q in queries.split(chunk):
        a, b, c = (tri[None, :, k] - q[:, None] for k in range(3))   # (n,F,3)
        la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
        det = (a * torch.linalg.cross(b, c)).sum(-1)
        den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
        omega = 2 * torch.atan2(det, den)                    # signed solid angle
        out.append(omega.sum(-1) / (4 * math.pi))
    return torch.cat(out)


@torch.no_grad()
def inside_vertices(verts, faces, thresh: float = 1.0, ignore_mask=None):
    """Mask (V,) of vertices that lie inside the body. ignore_mask: vertices not to count."""
    w = winding_numbers(verts, faces)
    m = w > thresh
    if ignore_mask is not None:
        m &= ~ignore_mask
    return m


@torch.no_grad()
def count_collisions(verts_batch, faces, **kw):
    """(B,V,3) -> (B,) number of vertices inside the body."""
    return torch.stack([inside_vertices(v, faces, **kw).sum() for v in verts_batch])


def collision_rate(counts, C: int = 0):
    """Col.Rate@C (%): fraction of frames with count > C."""
    counts = torch.as_tensor(counts)
    return 100.0 * (counts > C).float().mean().item()
