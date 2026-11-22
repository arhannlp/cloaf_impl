"""
Gerçek SMPL dosyası OLMADAN kodun uçtan uca çalıştığını test eder.
Küre şeklinde sahte bir "SMPL" modeli oluşturur (sayılar anlamsızdır!).

  python tests/smoke_test.py
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
    v, f = icosphere(3)
    v = v * np.array([0.25, 0.8, 0.15])                     # uzun elipsoid "vücut"
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


def main():
    from cloaf.body import SMPLBody, pinv_solve
    from cloaf.collision import count_collisions, winding_numbers
    from cloaf.model import MotionField
    from cloaf.train import trajectory_loss
    from cloaf.integrate import cloaf_solve, correct_sequence

    # sahte mesh 6890 vertexten az: smplx'in ekstra keypoint seçicisini devre dışı bırak
    import smplx.vertex_joint_selector as vjs
    vjs.VertexJointSelector.forward = lambda self, vertices, joints: joints

    tmp = tempfile.mkdtemp()
    make_fake_smpl(tmp)
    body = SMPLBody(tmp, device="cpu", exclude_radius=0.0)
    torch.manual_seed(0)

    # 1) Jacobian: fd vs jacfwd ve lineerlik
    th0 = 0.1 * torch.randn(2, 69, dtype=torch.float64)
    sample = body.sample_surface(300)
    J_fd = body.jacobian(th0, sample, method="fd")
    try:
        J_ad = body.jacobian(th0, sample, method="jacfwd")
        print("jacobian fd vs jacfwd max fark:", (J_fd - J_ad).abs().max().item())
    except Exception as e:  # noqa: BLE001
        print("jacfwd çalışmadı (fd kullan):", type(e).__name__, e)
    d = 1e-4 * torch.randn(2, 69, dtype=torch.float64)
    f = (body.points(th0 + d, sample) - body.points(th0, sample)).reshape(2, -1)
    print("J d ≈ f göreli hata:", ((J_fd @ d[..., None])[..., 0] - f).norm().item() / f.norm().item())
    print("pinv RE:", ((pinv_solve(J_fd, f) - d).norm() / d.norm()).item(),
          "(sahte modelde çok eklem etkisiz olabilir -> yüksek olabilir)")

    # 2) GWN: yüzey vertexleri ~0.5, iç nokta ~1
    v = body.vertices(th0[:1])[0]
    w = winding_numbers(v.float(), body.faces, torch.zeros(1, 3))
    ws = winding_numbers(v.float(), body.faces)
    print(f"GWN merkez {w.item():.3f} (≈1), yüzey ort {ws.mean():.3f} (≈0.5)")
    print("çakışma sayısı:", count_collisions(v[None], body.faces).tolist())

    # 3) eğitim adımı
    net = MotionField()
    opt = torch.optim.Adam(net.parameters(), 1e-4)
    th1 = th0 + 0.05 * torch.randn_like(th0)
    for it in range(3):
        loss = trajectory_loss(body, net, th0, th1, S=200, M=4)
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"train it {it} loss {loss.item():.3e}")

    # 4) ODE çözümü + dizi düzeltme
    traj, nfe = cloaf_solve(body, net, th0[:1], th1[:1], S=200, rtol=1e-3, atol=1e-4)
    print("ODE nfe:", nfe, "| son pozun hedefe uzaklığı:", (traj[-1] - th1[:1]).norm().item())
    seq = th0[:1] + torch.cumsum(0.02 * torch.randn(4, 69, dtype=torch.float64), 0)
    out, counts = correct_sequence(body, net, seq, S=200, mode="colliding_only",
                                   rtol=1e-3, atol=1e-4)
    print("dizi düzeltme tamam:", out.shape, counts.tolist())
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
