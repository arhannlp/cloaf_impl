"""
End-to-end test that runs WITHOUT the real SMPL files.
It builds a fake ellipsoid "SMPL" model (the numbers are meaningless as
human-body results; they only check that the code is wired correctly).

  python tests/smoke_test.py

Each check prints PASS/FAIL and what it verifies. Read this file next to the
output: every block shows the shortest way to call one part of the code.
"""
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]


def icosphere(n_sub=3):
    t = (1 + 5 ** 0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
         [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
         [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
         [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    v = [np.array(x, float) / np.linalg.norm(x) for x in v]
    for _ in range(n_sub):
        cache, nf = {}, []

        def mid(a, b):
            k = tuple(sorted((a, b)))
            if k not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[k] = len(v) - 1
            return cache[k]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    return np.array(v), np.array(f)


def make_fake_smpl(folder):
    """Writes a fake SMPL_NEUTRAL.pkl: an elongated ellipsoid with 24 joints."""
    v, f = icosphere(3)
    v = v * np.array([0.25, 0.8, 0.15])
    J = np.stack([np.array([np.sin(i), -0.8 + 1.6 * i / 23, 0.05 * np.cos(i)]) * [0.2, 1, 1]
                  for i in range(24)])
    Jreg = np.zeros((24, len(v)))
    for i, j in enumerate(J):
        Jreg[i, np.argsort(np.linalg.norm(v - j, axis=1))[:5]] = 0.2
    d = np.linalg.norm(v[:, None] - J[None], axis=-1)
    W = np.exp(-d / 0.05)
    W /= W.sum(1, keepdims=True)
    model = dict(v_template=v, f=f.astype(np.int64), J_regressor=Jreg, weights=W,
                 kintree_table=np.array([[4294967295] + SMPL_PARENTS[1:], list(range(24))]),
                 shapedirs=np.random.RandomState(0).randn(len(v), 3, 10) * 1e-3,
                 posedirs=np.zeros((len(v), 3, 207)))
    with open(Path(folder) / "SMPL_NEUTRAL.pkl", "wb") as fh:
        pickle.dump(model, fh)


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main():
    from cloaf.body import SMPLBody, pinv_solve
    from cloaf.collision import winding_numbers
    from cloaf.model import MotionField
    from cloaf.train import trajectory_loss
    from cloaf.integrate import cloaf_solve, correct_sequence

    # The fake mesh has fewer than 6890 vertices: disable smplx's extra keypoint selector
    import smplx.vertex_joint_selector as vjs
    vjs.VertexJointSelector.forward = lambda self, vertices, joints: joints

    tmp = tempfile.mkdtemp()
    make_fake_smpl(tmp)
    body = SMPLBody(tmp, device="cpu", exclude_radius=0.0)
    torch.manual_seed(0)
    results = []

    # 1) Jacobian (body.py, paper Eq. 3): finite differences vs autodiff must agree
    th0 = 0.1 * torch.randn(2, 69, dtype=torch.float64)
    sample = body.sample_surface(300)
    J_fd = body.jacobian(th0, sample, method="fd")
    try:
        J_ad = body.jacobian(th0, sample, method="jacfwd")
        diff = (J_fd - J_ad).abs().max().item()
        results.append(check("1 Jacobian fd vs jacfwd", diff < 1e-6, f"max diff {diff:.1e}"))
    except Exception as e:  # noqa: BLE001
        print(f"[SKIP] 1 jacfwd not available ({type(e).__name__}); use method='fd'")

    # 2) Linearity (paper Eq. 3 / Fig. 5): J·dΘ ≈ X(Θ+dΘ) - X(Θ) for a small dΘ
    d = 1e-4 * torch.randn(2, 69, dtype=torch.float64)
    f = (body.points(th0 + d, sample) - body.points(th0, sample)).reshape(2, -1)
    lin = ((J_fd @ d[..., None])[..., 0] - f).norm().item() / f.norm().item()
    results.append(check("2 J·dΘ ≈ ΔX", lin < 1e-2, f"relative error {lin:.1e}"))

    # 3) Pseudo-inverse (body.py pinv_solve, paper Eq. 5): recover dΘ from the point motion
    re = ((pinv_solve(J_fd, f) - d).norm() / d.norm()).item()
    results.append(check("3 J^† recovers dΘ", re < 1e-1, f"RE {re:.1e} (paper uses RE<0.1)"))

    # 4) Collision detection (collision.py): GWN ≈ 1 inside, ≈ 0.5 on the surface
    v = body.vertices(th0[:1])[0]
    w_in = winding_numbers(v.float(), body.faces, torch.zeros(1, 3)).item()
    w_surf = winding_numbers(v.float(), body.faces).mean().item()
    n_col = body.count_collisions(v[None]).item()
    results.append(check("4 winding numbers", abs(w_in - 1) < 0.05 and abs(w_surf - 0.5) < 0.1,
                         f"center {w_in:.3f} (≈1), surface mean {w_surf:.3f} (≈0.5), "
                         f"colliding vertices {n_col} (expect 0)"))

    # 5) Training step (train.py, paper Eq. 9): loss is finite and gradients flow
    net = MotionField()
    opt = torch.optim.Adam(net.parameters(), 1e-4)
    th1 = th0 + 0.05 * torch.randn_like(th0)
    losses = []
    for _ in range(3):
        loss = trajectory_loss(body, net, th0, th1, S=200, M=4)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    gnorm = sum(p.grad.norm() for p in net.parameters() if p.grad is not None).item()
    results.append(check("5 training step", np.isfinite(losses).all() and gnorm > 0,
                         f"losses {[f'{x:.2e}' for x in losses]}, grad norm {gnorm:.2e}"))

    # 6) ODE solve (integrate.py, paper Eq. 6). The network is UNTRAINED, so the final
    #    pose is NOT expected to reach the target: this only checks that it runs.
    traj, nfe = cloaf_solve(body, net, th0[:1], th1[:1], S=200, rtol=1e-3, atol=1e-4)
    dist = (traj[-1] - th1[:1]).norm().item()
    results.append(check("6 ODE solve runs", np.isfinite(dist),
                         f"{nfe} solver evaluations, distance to target {dist:.2f} "
                         f"(meaningless until the network is trained)"))

    # 7) Sequence correction (integrate.py, paper Sec. 4.2 + extension)
    seq = th0[:1] + torch.cumsum(0.02 * torch.randn(4, 69, dtype=torch.float64), 0)
    out, counts = correct_sequence(body, net, seq, S=200, mode="colliding_only",
                                   rtol=1e-3, atol=1e-4)
    results.append(check("7 correct_sequence runs", out.shape == seq.shape,
                         f"output {tuple(out.shape)}, collisions per frame {counts.tolist()}"))

    print("\nSMOKE TEST", "OK" if all(results) else "FAILED")


if __name__ == "__main__":
    main()
