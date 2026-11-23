"""
Motion field network f_ω(x | Θ(t), Θ1, Δt)  (Eq. 8).

Known from the paper:
  - A variant of ShapeFlow [15], 6-layer MLP; the output of every layer except
    the last is concatenated with the input vector.
  - Fourier features on all inputs, n_f = 20 frequencies.
  - "Time left" Δt = t1 - t instead of the absolute time t.
  - The output is scaled by the expected speed averaged over the points.

AMBIGUOUS in the paper (discuss with the supervisor / report as assumptions):
  - Hidden width (256 here) and activation (LeakyReLU).
  - Whether the network is per-point or takes all points as one vector.
    Here it is PER-POINT like ShapeFlow: for each point [FF(x_i), z] -> v_i (3),
    with a per-body code z = Enc(FF(Θ(t)), FF(Θ1), FF(Δt)). This is Lipschitz in x.
"""
import math
import torch
import torch.nn as nn


DT_MIN = 0.1   # smallest "time left" used to scale velocities (see MotionField.forward)


class FourierFeatures(nn.Module):
    def __init__(self, n_freq: int = 20, max_log2: float = 6.0):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.linspace(0, max_log2, n_freq)) * math.pi)

    def out_dim(self, d):
        return d * (1 + 2 * len(self.freqs))

    def forward(self, x):
        xf = x[..., None] * self.freqs                    # (...,d,n)
        return torch.cat([x, xf.sin().flatten(-2), xf.cos().flatten(-2)], -1)


class DenseConcatMLP(nn.Module):
    """6 layers; every hidden output is concatenated with the input (ShapeFlow ImNet style)."""
    def __init__(self, d_in, d_out, hidden=256, n_layers=6):
        super().__init__()
        self.layers = nn.ModuleList()
        d = d_in
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(d, hidden))
            d = hidden + d_in
        self.out = nn.Linear(d, d_out)
        self.act = nn.LeakyReLU(0.02)

    def forward(self, x):
        h = x
        for lin in self.layers:
            h = torch.cat([self.act(lin(h)), x], -1)
        return self.out(h)


class MotionField(nn.Module):
    def __init__(self, pose_dim=69, n_freq=20, hidden=256, code_dim=256):
        super().__init__()
        self.ff = FourierFeatures(n_freq)
        d_body = self.ff.out_dim(2 * pose_dim + 1)
        self.body_enc = nn.Sequential(nn.Linear(d_body, code_dim), nn.LeakyReLU(0.02),
                                      nn.Linear(code_dim, code_dim))
        self.point_net = DenseConcatMLP(self.ff.out_dim(3) + code_dim, 3, hidden, 6)
        nn.init.zeros_(self.point_net.out.bias)

    def forward(self, x, theta_t, theta_1, dt, x_target=None):
        """
        x        : (B,S,3) current points X(Θ(t))
        theta_t  : (B,69), theta_1 : (B,69), dt : (B,) time left
        x_target : (B,S,3) X(Θ1) -> used for the speed scale
        returns  : (B,S,3) velocities dX/dt
        """
        B, S, _ = x.shape
        z = self.body_enc(self.ff(torch.cat([theta_t, theta_1, dt[:, None]], -1)))
        h = torch.cat([self.ff(x), z[:, None].expand(B, S, -1)], -1)
        v = self.point_net(h)
        if x_target is not None:
            # "expected speed": remaining distance / remaining time, averaged over points
            # dt >= DT_MIN: training only sees dt >= 1/M = 0.1, while an ODE solver also
            # evaluates the field at t = 1 (dt = 0), where the speed would blow up
            speed = (x_target - x).norm(dim=-1).mean(-1) / dt.clamp_min(DT_MIN)   # (B,)
            v = v * speed[:, None, None]
        return v
