"""
Put the training loss in context: compare a checkpoint with two reference fields
on the same random pose pairs.

  zero field   : the body does not move at all       -> the loss a useless network gets
  point target : each point moves straight to its own target position (Supp. C)
                 -> roughly what a well-trained network should reach
  network      : the checkpoint

  python scripts/loss_reference.py --smpl data/body_models/smpl \
      --amass data/amass_smpl_30fps.npz --field checkpoints/field.pt

A network that has learned something has a loss clearly below the zero field,
ideally close to the point-target field.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloaf.body import SMPLBody  # noqa: E402
from cloaf.integrate import PointTargetField  # noqa: E402
from cloaf.model import MotionField  # noqa: E402
from cloaf.train import PairSampler, trajectory_loss  # noqa: E402


class ZeroField(torch.nn.Module):
    def forward(self, x, theta_t, theta_1, dt, x_target=None):
        return torch.zeros_like(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl", required=True)
    ap.add_argument("--amass", required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--M", type=int, default=10)
    ap.add_argument("--S", type=int, default=1000)
    ap.add_argument("--max_gap", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    body = SMPLBody(a.smpl, device=a.device, dtype=torch.float32)
    net = MotionField().to(a.device).eval()
    net.load_state_dict(torch.load(a.field, map_location=a.device))
    fields = {"zero field": ZeroField(), "point target": PointTargetField(), "network": net}
    sampler = PairSampler(a.amass, max_gap=a.max_gap, device=a.device)

    torch.manual_seed(0)
    sums = {k: 0.0 for k in fields}
    with torch.no_grad():
        for _ in range(a.batches):
            th0, th1 = sampler(a.batch)
            seed = torch.randint(0, 2 ** 31, (1,)).item()
            for k, f in fields.items():
                torch.manual_seed(seed)            # same surface sample for every field
                sums[k] += trajectory_loss(body, f, th0, th1, a.S, a.M, jac_method="jacfwd").item()
    ref = sums["zero field"]
    print(f"{'field':14s} {'loss':>10s} {'vs zero field':>14s}")
    for k, v in sums.items():
        print(f"{k:14s} {v / a.batches:10.3e} {v / ref:13.2f}x")


if __name__ == "__main__":
    main()
