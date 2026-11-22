"""
3DPW evaluation metrics (Sec. 4.1): MPJPE, P-MPJPE (PA-MPJPE), Accel.Err, Col.Rate.
Standard 3DPW protocol (same as SPIN / VIBE / HMR2.0):
  - 14 LSP joints from the H36M regressor (J_regressor_h36m.npy, SPIN data package)
  - Pelvis (mid-hip) alignment, in mm
  - GT: gendered SMPL; prediction: neutral SMPL
"""
import torch

H36M_TO_J14 = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 10]


def joints14(verts, J_regressor_h36m):
    """verts (N,6890,3), J_reg (17,6890) -> (N,14,3)"""
    J = torch.einsum("jv,nvc->njc", J_regressor_h36m.to(verts), verts)
    return J[:, H36M_TO_J14]


def _pelvis_align(j):
    return j - 0.5 * (j[:, 2:3] + j[:, 3:4])   # LSP: 2=r_hip, 3=l_hip


def mpjpe(pred, gt):
    return 1000 * (_pelvis_align(pred) - _pelvis_align(gt)).norm(dim=-1).mean().item()


def procrustes(S1, S2):
    """Align S1 to S2 with a similarity transform (batched). (N,J,3)"""
    mu1, mu2 = S1.mean(1, keepdim=True), S2.mean(1, keepdim=True)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = (X1 ** 2).sum((1, 2))
    K = X1.transpose(1, 2) @ X2
    U, s, Vh = torch.linalg.svd(K)
    Z = torch.eye(3, device=S1.device, dtype=S1.dtype).repeat(len(S1), 1, 1)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ Vh))
    R = Vh.transpose(1, 2) @ Z @ U.transpose(1, 2)
    scale = torch.einsum("nii->n", R @ K) / var1
    return scale[:, None, None] * (X1 @ R.transpose(1, 2)) + mu2


def pa_mpjpe(pred, gt):
    return 1000 * (procrustes(pred, gt) - gt).norm(dim=-1).mean().item()


def accel_error(pred, gt):
    """compute_error_accel from VIBE: difference of second differences (mm/frame^2).
    The paper writes mm/s^2, but the literature (VIBE, TCMR) uses this same formula.
    Careful: only compute over consecutive valid frames."""
    p, g = _pelvis_align(pred), _pelvis_align(gt)
    ap = p[:-2] - 2 * p[1:-1] + p[2:]
    ag = g[:-2] - 2 * g[1:-1] + g[2:]
    return 1000 * (ap - ag).norm(dim=-1).mean().item()


def temporal_jerk(poses_or_joints):
    """For the extension: measure sudden jumps at transitions between corrected
    and uncorrected frames (third difference along time)."""
    x = poses_or_joints
    return (x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]).norm(dim=-1)
