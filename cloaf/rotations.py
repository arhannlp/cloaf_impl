"""Rotation conversions (no pytorch3d / roma dependency)."""
import torch


def rotmat_to_aa(R):
    """(...,3,3) rotation matrices -> (...,3) axis-angle, via quaternions (robust near 0 and pi)."""
    R = R.double()
    m = R.reshape(-1, 3, 3)
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.zeros(len(m), 4, dtype=m.dtype, device=m.device)       # (w, x, y, z)
    c0 = t > 0
    s = torch.sqrt(t[c0] + 1.0) * 2
    q[c0] = torch.stack([0.25 * s, (m[c0, 2, 1] - m[c0, 1, 2]) / s,
                         (m[c0, 0, 2] - m[c0, 2, 0]) / s, (m[c0, 1, 0] - m[c0, 0, 1]) / s], -1)
    c1 = ~c0 & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    s = torch.sqrt(1.0 + m[c1, 0, 0] - m[c1, 1, 1] - m[c1, 2, 2]) * 2
    q[c1] = torch.stack([(m[c1, 2, 1] - m[c1, 1, 2]) / s, 0.25 * s,
                         (m[c1, 0, 1] + m[c1, 1, 0]) / s, (m[c1, 0, 2] + m[c1, 2, 0]) / s], -1)
    c2 = ~c0 & ~c1 & (m[:, 1, 1] > m[:, 2, 2])
    s = torch.sqrt(1.0 + m[c2, 1, 1] - m[c2, 0, 0] - m[c2, 2, 2]) * 2
    q[c2] = torch.stack([(m[c2, 0, 2] - m[c2, 2, 0]) / s, (m[c2, 0, 1] + m[c2, 1, 0]) / s,
                         0.25 * s, (m[c2, 1, 2] + m[c2, 2, 1]) / s], -1)
    c3 = ~c0 & ~c1 & ~c2
    s = torch.sqrt(1.0 + m[c3, 2, 2] - m[c3, 0, 0] - m[c3, 1, 1]) * 2
    q[c3] = torch.stack([(m[c3, 1, 0] - m[c3, 0, 1]) / s, (m[c3, 0, 2] + m[c3, 2, 0]) / s,
                         (m[c3, 1, 2] + m[c3, 2, 1]) / s, 0.25 * s], -1)
    q = q * torch.sign(q[:, :1] + 1e-12)                                 # w >= 0
    v = q[:, 1:]
    sin_half = v.norm(dim=-1).clamp_min(1e-12)
    angle = 2 * torch.atan2(sin_half, q[:, 0])
    return (v / sin_half[:, None] * angle[:, None]).reshape(*R.shape[:-2], 3)


def aa_to_rotmat(aa):
    """(...,3) axis-angle -> (...,3,3) rotation matrices."""
    from smplx.lbs import batch_rodrigues
    return batch_rodrigues(aa.reshape(-1, 3)).reshape(*aa.shape[:-1], 3, 3)
