"""
Self-intersection detection with the Generalized Winding Number (GWN).

For a vertex on a closed surface the GWN is ≈ 0.5 (it lies on its own surface).
If the vertex is INSIDE another body part, the GWN is ≈ 1.5.

Two tests are available (SMPLBody(collision_mode=...)):

  "parts" (default): a vertex counts only if it is inside a body part that is
     NOT the same as or adjacent to its own part in the kinematic tree (e.g. a
     hand inside the torso, a forearm inside the other arm, one leg inside the
     other). Contact between adjacent parts (upper arm against torso when the arms
     hang down, forearm against upper arm) is ignored, as in SMPLify-X / TUCH.
     On real AMASS motion the "full" test flags hundreds of such vertices in
     almost every frame, which would make Col.Rate ~100 %.
  "full": a vertex counts if it is inside the whole body (GWN of the full mesh > 1).

Paper metric (Sec. 4.1): Col.Rate@C = percentage of samples (frames) with
more than C vertices inside the body. The paper does NOT describe how this is
detected; "parts" is our assumption. Calibrate against Table 1 (HMR2.0 39.2 %)
when 3DPW predictions are available.

GWN is computed in float32 and re-computed in float64 for the vertices whose
value is close to the threshold (float32 is unreliable for near-touching surfaces).
"""
import math
import torch

# SMPL joint -> body part (by the dominant skinning weight of a vertex)
PART_NAMES = ["torso", "head", "l_upper_arm", "l_forearm", "l_hand", "r_upper_arm", "r_forearm",
              "r_hand", "l_thigh", "l_shin", "l_foot", "r_thigh", "r_shin", "r_foot"]
JOINT_TO_PART = [0, 8, 11, 0, 9, 12, 0, 10, 13, 0, 10, 13, 1, 0, 0, 1, 2, 5, 3, 6, 4, 7, 4, 7]
PART_ADJACENT = [(0, 1), (0, 2), (0, 5), (0, 8), (0, 11), (2, 3), (3, 4), (5, 6), (6, 7),
                 (8, 9), (9, 10), (11, 12), (12, 13)]


def winding_numbers(verts, faces, queries=None, chunk: int = 512, dtype=torch.float32):
    """verts (V,3), faces (F,3), queries (N,3) -> (N,) GWN. Run on GPU for speed."""
    verts = verts.to(dtype)
    queries = verts if queries is None else queries.to(dtype)
    tri = verts[faces]                                       # (F,3,3)
    out = []
    for q in queries.split(chunk):
        a, b, c = (tri[None, :, k] - q[:, None] for k in range(3))   # (n,F,3)
        la, lb, lc = a.norm(dim=-1), b.norm(dim=-1), c.norm(dim=-1)
        det = (a * torch.linalg.cross(b, c, dim=-1)).sum(-1)
        den = la * lb * lc + (a * b).sum(-1) * lc + (b * c).sum(-1) * la + (c * a).sum(-1) * lb
        omega = 2 * torch.atan2(det, den)                    # signed solid angle
        out.append(omega.sum(-1) / (4 * math.pi))
    if not out:
        return torch.zeros(0, dtype=dtype, device=verts.device)
    return torch.cat(out)


def _above(verts, faces, queries, thresh, margin=0.25):
    """GWN(queries) > thresh, float32 with a float64 re-check near the threshold."""
    w = winding_numbers(verts, faces, queries)
    amb = (w - thresh).abs() < margin
    if amb.any():
        w = w.double()
        w[amb] = winding_numbers(verts, faces, queries[amb], chunk=256, dtype=torch.float64)
    return w > thresh


@torch.no_grad()
def inside_vertices(verts, faces, thresh: float = 1.0, ignore_mask=None):
    """"full" test: mask (V,) of vertices inside the whole body."""
    m = _above(verts, faces, verts, thresh)
    if ignore_mask is not None:
        m &= ~ignore_mask
    return m


class PartCollisionTest:
    """"parts" test: a vertex collides if it is inside a non-adjacent body part.
    The GWN of a part's (open) surface is ≈ 1 deep inside it, so thresh = 0.6."""

    def __init__(self, lbs_weights, faces, thresh: float = 0.6):
        jp = torch.as_tensor(JOINT_TO_PART, device=faces.device)
        self.vertex_part = jp[lbs_weights.argmax(-1).to(faces.device)]      # (V,)
        face_parts = self.vertex_part[faces]                                  # (F,3)
        n = len(PART_NAMES)
        near = torch.eye(n, dtype=torch.bool, device=faces.device)
        for a, b in PART_ADJACENT:
            near[a, b] = near[b, a] = True
        self.near = near                                   # (n_parts, n_parts): same or adjacent
        self.thresh = thresh
        self.groups = []
        for p in range(n):
            idx = (self.vertex_part == p).nonzero().squeeze(-1)
            far = ~near[p][face_parts].any(-1)            # faces with no vertex in p or its neighbours
            self.groups.append((idx, faces[far]))

    @torch.no_grad()
    def __call__(self, verts):
        m = torch.zeros(verts.shape[0], dtype=torch.bool, device=verts.device)
        for idx, far_faces in self.groups:
            if len(idx) and len(far_faces):
                m[idx] = _above(verts, far_faces, verts[idx], self.thresh)
        return m


@torch.no_grad()
def count_collisions(verts_batch, faces, ignore_mask=None, test=None):
    """(B,V,3) -> (B,) number of colliding vertices. test: a PartCollisionTest or None (full)."""
    counts = []
    for v in verts_batch:
        m = test(v) if test is not None else inside_vertices(v, faces)
        if ignore_mask is not None:
            m &= ~ignore_mask
        counts.append(m.sum())
    return torch.stack(counts)


def collision_rate(counts, C: int = 0):
    """Col.Rate@C (%): fraction of frames with count > C."""
    counts = torch.as_tensor(counts)
    return 100.0 * (counts > C).float().mean().item()
