"""
Hareket alanı ağı f_ω(x | Θ(t), Θ1, Δt)  (Denk. 8).

Makaleden bilinenler:
  - ShapeFlow [15] varyantı, 6 katmanlı MLP; son hariç her katmanın çıktısı
    giriş vektörüyle birleştirilir (concat).
  - Tüm girişlere Fourier features, n_f = 20 frekans.
  - Mutlak zaman t yerine "kalan zaman" Δt = t1 - t.
  - Çıktı, noktalar üzerinden ortalama "beklenen hız" ile ölçeklenir.

Makalede BELİRSİZ olanlar (supervisor ile konuş / varsayım olarak raporla):
  - Gizli katman genişliği (burada 256), aktivasyon (LeakyReLU).
  - Ağın nokta-bazlı mı yoksa tüm noktaları tek vektör olarak mı aldığı.
    Burada ShapeFlow gibi NOKTA-BAZLI: her nokta için [FF(x_i), z] -> v_i (3),
    z = vücut başına kod = Enc(FF(Θ(t)), FF(Θ1), FF(Δt)). Bu x'e göre Lipschitz.
"""
import math
import torch
import torch.nn as nn


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
    """6 katman; her gizli katman çıktısı girişle concat edilir (ShapeFlow ImNet tarzı)."""
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
        x        : (B,S,3) mevcut noktalar X(Θ(t))
        theta_t  : (B,69), theta_1 : (B,69), dt : (B,) kalan zaman
        x_target : (B,S,3) X(Θ1) -> hız ölçeği için
        dönüş    : (B,S,3) hızlar dX/dt
        """
        B, S, _ = x.shape
        z = self.body_enc(self.ff(torch.cat([theta_t, theta_1, dt[:, None]], -1)))
        h = torch.cat([self.ff(x), z[:, None].expand(B, S, -1)], -1)
        v = self.point_net(h)
        if x_target is not None:
            # "beklenen hız": kalan mesafe / kalan zaman, noktalar üzerinden ortalama
            speed = (x_target - x).norm(dim=-1).mean(-1) / dt.clamp_min(1e-3)   # (B,)
            v = v * speed[:, None, None]
        return v
