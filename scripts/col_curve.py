"""
Col.Rate@C as a function of C, from the per-frame collision counts saved by eval_3dpw.py --out.
Shows how our collision test relates to the paper's (Table 1: HMR2.0 Col@0 39.2 %, Col@100 21.6 %).

  python scripts/col_curve.py --eval results/eval_hmr2.npz --png results/col_curve.png
"""
import argparse

import numpy as np

PAPER = {0: 39.2, 100: 21.6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--png", default=None)
    a = ap.parse_args()
    d = np.load(a.eval, allow_pickle=True)
    counts = np.concatenate([d[k].item()["counts"] for k in d.files if k != "summary"])
    rate = lambda C: 100.0 * (counts > C).mean()
    print(f"{len(counts)} frames, median count {np.median(counts):.0f}, max {counts.max()}")
    print("   C   Col@C   paper")
    for C in [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300]:
        print(f"{C:4d}  {rate(C):5.1f} %  {'' if C not in PAPER else f'{PAPER[C]:.1f} %'}")
    Cs = np.arange(0, 301)
    r = np.array([rate(C) for C in Cs])
    c_match = int(Cs[np.argmin(np.abs(r - PAPER[0]))])
    print(f"our test gives the paper's Col@0 ({PAPER[0]} %) at C = {c_match} ({rate(c_match):.1f} %)")
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 3.5))
        plt.plot(Cs, r, label="HMR2.0, our collision test")
        plt.scatter(list(PAPER), list(PAPER.values()), c="red", zorder=3, label="HMR2.0, paper (Table 1)")
        plt.xlabel("C (colliding vertices)")
        plt.ylabel("Col.Rate@C (%)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(a.png, dpi=150)
        print(f"saved {a.png}")


if __name__ == "__main__":
    main()
