"""
Training the motion field network on AMASS (Sec. 3, "Network Architecture and Training").

Training tricks (the paper says all of them are needed, Supp. B):
  1. Use the linear interpolation Θ~(t) = Θ0 + t (Θ1-Θ0) instead of Θ(t)
     -> no ODE solve needed, an M-step cumulative sum is enough.
  2. Trajectory loss (Eq. 9): L = 1/M Σ_m || X(Θ_{0->t_m}) - X(Θ~(t_m)) ||^2
  3. Input Δt = t1 - t, output scaled by the average expected speed.
  4. Consecutive poses, Δt_max = 1.

Usage:
  python scripts/prepare_amass.py --amass_root data/amass --out data/amass_smpl_30fps.npz
  python -m cloaf.train --smpl data/body_models/smpl --amass data/amass_smpl_30fps.npz
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .body import SMPLBody, pinv_solve
from .model import MotionField


# ---------------------------------------------------------------------- #
def prepare_amass(amass_root, out_path, target_fps=30, max_files=None):
    """AMASS (SMPL+H npz) -> a single npz: body_pose (N,69), seq_id (N,).
    amass_root: one folder or a list of folders (e.g. ["data/amass/ACCAD", "data/amass/CMU"]).
    AMASS 'poses' is 156-dim (SMPL+H): [0:3] global, [3:66] 21 body joints,
    [66:] hand joints. SMPL joints 22-23 (hands) are set to zero."""
    import glob, os
    roots = [amass_root] if isinstance(amass_root, (str, os.PathLike)) else list(amass_root)
    poses, seq = [], []
    files = []
    for root in roots:
        found = sorted(glob.glob(os.path.join(root, "**/*.npz"), recursive=True))
        found = [f for f in found if not f.endswith("shape.npz")]
        if max_files is not None:
            found = found[:max_files]
        if not found:
            raise FileNotFoundError(f"no AMASS .npz files under {root}")
        print(f"{root}: {len(found)} files")
        files += found
    for i, f in enumerate(files):
        d = np.load(f)
        if "poses" not in d:
            continue
        fps = float(d["mocap_framerate"] if "mocap_framerate" in d else d["mocap_frame_rate"])
        step = max(1, int(round(fps / target_fps)))
        p = d["poses"][::step]
        body = np.concatenate([p[:, 3:66], np.zeros((len(p), 6))], 1)
        poses.append(body.astype(np.float32))
        seq.append(np.full(len(p), i, dtype=np.int32))
    # TODO: the paper's split is unknown; hold out a few sub-datasets for validation.
    np.savez(out_path, body_pose=np.concatenate(poses), seq_id=np.concatenate(seq))
    return int(sum(len(p) for p in poses))


class PairSampler:
    """(Θ0, Θ1) pairs from the same sequence, 1..max_gap frames apart."""
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
            # simple rejection sampling: drop pairs that cross a sequence boundary
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
            J = body.jacobian(th_lin, sample, method=jac_method)   # does not depend on ω
        dt = torch.full((B,), 1.0 - t, device=th0.device)
        f = net(x_lin.float(), th_lin.float(), th1.float(), dt, x_target=x1)
        dtheta = pinv_solve(J, f.reshape(B, -1))               # differentiable through f
        theta = theta + dtheta / M                             # Euler step
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
    ap.add_argument("--minutes", type=float, default=None,
                    help="stop after this many minutes and save (time budget)")
    ap.add_argument("--max_gap", type=int, default=4, help="max frame gap between Θ0 and Θ1")
    ap.add_argument("--precision", default="float32", choices=["float32", "float64"],
                    help="SMPL/Jacobian precision during training. The paper uses float64, but on a "
                         "T4 float64 is ~30x slower; float32 + forward-mode AD is accurate enough here")
    ap.add_argument("--jac", default=None, choices=["fd", "jacfwd"],
                    help="Jacobian method (default: jacfwd for float32, fd for float64)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>.state (network, optimizer, iteration)")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    state_path, log_path = a.out + ".state", a.out + ".log.csv"

    dtype = torch.float32 if a.precision == "float32" else torch.float64
    jac = a.jac or ("jacfwd" if a.precision == "float32" else "fd")
    if a.precision == "float32" and jac == "fd":
        print("warning: finite differences in float32 are inaccurate; use --jac jacfwd")
    body = SMPLBody(a.smpl, device=a.device, dtype=dtype)
    print(f"training with {a.precision}, Jacobian: {jac}, batch {a.batch}, M {a.M}, S {a.S}", flush=True)
    net = MotionField().to(a.device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sampler = PairSampler(a.amass, max_gap=a.max_gap, device=a.device)

    start = 0
    if a.resume and Path(state_path).exists():
        st = torch.load(state_path, map_location=a.device)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        start = st["it"] + 1
        print(f"resumed from {state_path} at iteration {start}")
    elif not Path(log_path).exists() or not a.resume:
        Path(log_path).write_text("iteration,loss,seconds\n")

    def save(it):
        torch.save(net.state_dict(), a.out)  # plain weights: what the scripts load
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "it": it}, state_path)

    t0 = time.time()
    running = []
    it = start
    for it in range(start, a.iters):
        th0, th1 = sampler(a.batch)
        loss = trajectory_loss(body, net, th0, th1, a.S, a.M, jac_method=jac)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        running.append(loss.item())
        if it - start < 100 and it % 10 == 0 and it > start:
            print(f"it {it:6d}  loss {loss.item():.3e}  ({time.time() - t0:.0f}s, "
                  f"{(time.time() - t0) / (it - start):.2f} s/it)", flush=True)
        if it % 100 == 0:
            avg = sum(running) / len(running)
            running = []
            print(f"it {it:6d}  loss(avg100) {avg:.3e}  ({time.time() - t0:.0f}s)", flush=True)
            with open(log_path, "a") as fh:
                fh.write(f"{it},{avg:.6e},{time.time() - t0:.0f}\n")
        if it % 1000 == 0:
            save(it)
        if a.minutes is not None and time.time() - t0 > 60 * a.minutes:
            print(f"time budget reached after {it + 1 - start} iterations")
            break
    save(it)
    print(f"saved {a.out} (iteration {it})")


if __name__ == "__main__":
    main()
