"""
Hareket alanı ağını AMASS üzerinde eğitme (Sec. 3, "Network Architecture and Training").

Eğitim hileleri (makalede hepsinin gerekli olduğu söyleniyor, Supp. B):
  1. Θ(t) yerine lineer interpolasyon Θ~(t) = Θ0 + t (Θ1-Θ0) kullan
     -> ODE çözmeye gerek kalmaz, M adımlık kümülatif toplam yeterli.
  2. Yörünge kaybı (Denk. 9): L = 1/M Σ_m || X(Θ_{0->t_m}) - X(Θ~(t_m)) ||^2
  3. Δt = t1 - t girdisi, çıktı ortalama beklenen hızla ölçekli.
  4. Ardışık pozlar, Δt_max = 1.

Kullanım:
  python -m cloaf.train --smpl data/body_models --amass data/amass_smpl_30fps.npz
"""
import argparse
import time

import numpy as np
import torch

from .body import SMPLBody, pinv_solve
from .model import MotionField


# ---------------------------------------------------------------------- #
def prepare_amass(amass_root, out_path, target_fps=30):
    """AMASS (SMPL+H npz) -> tek npz: body_pose (N,69), seq_id (N,).
    AMASS 'poses' 156 boyutlu (SMPL+H): [0:3] global, [3:66] 21 gövde eklemi,
    [66:] el eklemleri. SMPL'in 22-23 numaralı (el) eklemleri sıfır kabul edilir."""
    import glob, os
    poses, seq = [], []
    files = sorted(glob.glob(os.path.join(amass_root, "**/*_poses.npz"), recursive=True))
    for i, f in enumerate(files):
        d = np.load(f)
        if "poses" not in d:
            continue
        step = max(1, int(round(float(d["mocap_framerate"]) / target_fps)))
        p = d["poses"][::step]
        body = np.concatenate([p[:, 3:66], np.zeros((len(p), 6))], 1)
        poses.append(body.astype(np.float32))
        seq.append(np.full(len(p), i, dtype=np.int32))
    np.savez(out_path, body_pose=np.concatenate(poses), seq_id=np.concatenate(seq))
    # TODO: makalenin split'i bilinmiyor; 3DPW ile ilgili alt kümeleri ayırma gerekmez
    #       ama validation için birkaç alt veri setini dışarıda bırak.


class PairSampler:
    """Aynı diziden (Θ0, Θ1) çiftleri; aralarındaki frame farkı 1..max_gap."""
    def __init__(self, path, max_gap=4, device="cuda"):
        d = np.load(path)
        self.pose = torch.as_tensor(d["body_pose"], device=device)
        self.seq = torch.as_tensor(d["seq_id"], device=device)
        self.max_gap = max_gap

    def __call__(self, B):
        N = len(self.pose)
        while True:
            i = torch.randint(0, N - self.max_gap, (B,), device=self.pose.device)
            g = torch.randint(1, self.max_gap + 1, (B,), device=self.pose.device)
            ok = self.seq[i] == self.seq[i + g]
            if ok.all():
                return self.pose[i], self.pose[i + g]
            # basit reddetme örneklemesi
            i, g = i[ok], g[ok]
            if len(i) >= B // 2:
                return self.pose[i], self.pose[i + g]


# ---------------------------------------------------------------------- #
def trajectory_loss(body, net, th0, th1, S=1000, M=10, jac_method="fd"):
    B = th0.shape[0]
    sample = body.sample_surface(S)
    th0, th1 = th0.to(body.dtype), th1.to(body.dtype)
    with torch.no_grad():
        x1 = body.points(th1, sample).float()

    theta = th0                    # Θ_{0->t_0}
    loss = 0.0
    for m in range(M):
        t = m / M
        th_lin = th0 + t * (th1 - th0)                         # Θ~(t_m)
        with torch.no_grad():
            x_lin = body.points(th_lin, sample)
            J = body.jacobian(th_lin, sample, method=jac_method)   # ω'ya bağlı değil
        dt = torch.full((B,), 1.0 - t, device=th0.device)
        f = net(x_lin.float(), th_lin.float(), th1.float(), dt, x_target=x1)
        dtheta = pinv_solve(J, f.reshape(B, -1))               # türevlenebilir (f üzerinden)
        theta = theta + dtheta / M                             # Euler adımı
        th_lin_next = th0 + (m + 1) / M * (th1 - th0)
        x_pred = body.points(theta, sample)                    # X(Θ_{0->t_{m+1}})
        with torch.no_grad():
            x_ref = body.points(th_lin_next, sample)
        loss = loss + ((x_pred - x_ref) ** 2).sum(-1).mean()
    return loss / M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl", required=True)
    ap.add_argument("--amass", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--S", type=int, default=1000)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200_000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="checkpoints/field.pt")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    body = SMPLBody(a.smpl, device=a.device)
    net = MotionField().to(a.device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sampler = PairSampler(a.amass, device=a.device)

    t0 = time.time()
    for it in range(a.iters):
        th0, th1 = sampler(a.batch)
        loss = trajectory_loss(body, net, th0, th1, a.S, a.M)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if it % 100 == 0:
            print(f"it {it:6d}  loss {loss.item():.3e}  ({time.time() - t0:.0f}s)", flush=True)
        if it % 5000 == 0:
            torch.save(net.state_dict(), a.out)


if __name__ == "__main__":
    main()
