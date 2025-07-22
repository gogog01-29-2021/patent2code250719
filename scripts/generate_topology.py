#!/usr/bin/env python
"""
Minimal SIMP driver (placeholder): refine for production.
"""
import json, numpy as np

def load_spec(path):
    spec = json.load(open(path))
    # expect: loads, supports, material: {E, nu}, volume_fraction
    return spec

def run_simp(spec, nelx=60, nely=40, vol_frac=0.3, penal=3.0, iters=60):
    rho = vol_frac * np.ones((nely, nelx))
    history = []
    for k in range(iters):
        # FEM solve (placeholder)
        compliance = np.random.uniform(10,12) * (1 + 0.01*np.random.randn())
        # Sensitivity dummy
        dc = -np.ones_like(rho)
        # OC update (very simplified)
        move = 0.05
        rho = np.clip(rho + move * np.sign(dc), 0.01, 1.0)
        history.append((k, float(compliance), float(rho.mean())))
    return rho, history

if __name__ == "__main__":
    import argparse, csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--volume-fraction", type=float, default=0.3)
    ap.add_argument("--out-density", default="build/opt_density.npy")
    ap.add_argument("--log", default="logs/opt_curve.csv")
    args = ap.parse_args()

    spec = load_spec(args.spec)
    rho, hist = run_simp(spec, vol_frac=args.volume_fraction)
    np.save(args.out_density, rho)
    with open(args.log, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iter","compliance","rho_mean"])
        w.writerows(hist)
    print(f"[OK] density field → {args.out_density}; log → {args.log}")
 