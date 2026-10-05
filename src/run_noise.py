# -*- coding: utf-8 -*-
"""
============================================================================
 run_noise.py (v2 — complet)
 Robustesse aux donnees eparses et bruitees (critere 10).
 v2 : Adam 10000 + L-BFGS x2 (restarts) — coherence methodologique avec
      run_micromix/run_inverse ; l'erreur de base (~1.3%) permet enfin
      de MESURER l'effet du bruit (v1 : Adam seul -> base ~2%, effet noye).
 Duree : ~10-12 min par niveau (4 niveaux).
 Sorties : bruit.json + figure_bruit.png
============================================================================
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("DDE_BACKEND", "tensorflow")
import deepxde as dde

# ═══════════════ PARAMETRES ═══════════════
L, H, D, U, KAPPA = 5.0e-4, 1.0e-4, 1.0e-9, 1.0e-4, 8.0
Ls, DELTA, PE, Ts = L / H, 0.1, U * H / D, 1.0
NOISE = [0.0, 0.01, 0.05, 0.10]
NOBS, ITER, LBFGS_ITER, RESTARTS = 200, 10000, 2000, 2

def sig(x):
    return 1.0 / (1.0 + np.exp(-(np.asarray(x) - 0.5) / DELTA))

def solve_fdm(kappa, nx=250, ny=25, Ts=Ts):
    dx, dy = Ls / (nx - 1), 1.0 / (ny - 1)
    dt = min(0.2 * dy * dy / kappa, 0.2 * dx * dx, 0.4 * dx / (PE * 1.5))
    nt = max(50, int(np.ceil(Ts / dt)))
    dt = Ts / nt
    rx, ry = dt / (dx * dx), kappa * dt / (dy * dy)
    x = np.linspace(0, Ls, nx); y = np.linspace(0, 1, ny)
    u = 6.0 * y * (1.0 - y)
    X, Y = np.meshgrid(x, y)
    C = sig(Y); inlet = sig(y)
    for n in range(nt):
        Cn = C.copy(); core = Cn[1:-1, 1:-1]
        adv = dt * PE * u[1:-1, None] * (core - Cn[1:-1, :-2]) / dx
        C[1:-1, 1:-1] = core - adv \
            + rx * (Cn[1:-1, 2:] - 2.0 * core + Cn[1:-1, :-2]) \
            + ry * (Cn[2:, 1:-1] - 2.0 * core + Cn[:-2, 1:-1])
        C[:, 0] = inlet; C[:, -1] = C[:, -2]
        C[0, :] = C[1, :]; C[-1, :] = C[-2, :]
    return x, y, C

def L_mix_interp(x, C, frac=0.05):
    v = C.var(axis=0); thr = frac * v[0]
    idx = np.where(v <= thr)[0]
    if not len(idx): return float(x[-1])
    k = idx[0]
    if k == 0: return float(x[0])
    v1, v2 = v[k - 1], v[k]
    return float(x[k] if v1 == v2 else
                 x[k - 1] + (v1 - thr) / (v1 - v2) * (x[k] - x[k - 1]))

# ═══════════════ OBSERVATIONS ═══════════════
x, y, Cr = solve_fdm(KAPPA)
Xg, Yg = np.meshgrid(x, y)
rng = np.random.default_rng(42)
pts = np.column_stack([rng.uniform(0, Ls, NOBS), rng.uniform(0, 1, NOBS)])
from numpy import searchsorted
i0 = np.clip(searchsorted(x, pts[:, 0]) - 1, 0, len(x) - 2)
j0 = np.clip(searchsorted(y, pts[:, 1]) - 1, 0, len(y) - 2)
tx = (pts[:, 0] - x[i0]) / (x[i0 + 1] - x[i0])
ty = (pts[:, 1] - y[j0]) / (y[j0 + 1] - y[j0])
cobs = (Cr[j0, i0] * (1 - tx) * (1 - ty) + Cr[j0, i0 + 1] * tx * (1 - ty)
        + Cr[j0 + 1, i0] * (1 - tx) * ty + Cr[j0 + 1, i0 + 1] * tx * ty)

# ═══════════════ ENTRAINEMENTS ═══════════════
rows = []
for sn in NOISE:
    print("=" * 60)
    print("Bruit sigma = {:.0%} ({} observations)".format(sn, NOBS))
    yobs = (cobs + sn * rng.standard_normal(NOBS)).reshape(-1, 1)
    try:
        import tensorflow as tf
        tf.random.set_seed(42)
    except Exception:
        pass
    try:
        dde.config.set_random_seed(42)
    except Exception:
        pass
    geom = dde.geometry.Rectangle([0.0, 0.0], [Ls, 1.0])
    def pde(x_, y_):
        dc_x  = dde.grad.jacobian(y_, x_, i=0, j=0)
        dc_xx = dde.grad.hessian(y_, x_, i=0, j=0)
        dc_yy = dde.grad.hessian(y_, x_, i=1, j=1)     # CORRECTIF (i=1, j=1)
        uv = 6.0 * x_[:, 1:2] * (1.0 - x_[:, 1:2])
        return PE * uv * dc_x - dc_xx - KAPPA * dc_yy
    on_i = lambda x_, ob: ob and np.isclose(x_[0], 0.0)
    on_r = lambda x_, ob: ob and not np.isclose(x_[0], 0.0)
    try:
        NB, DB, PS = dde.icbc.NeumannBC, dde.icbc.DirichletBC, dde.icbc.PointSetBC
    except AttributeError:
        NB, DB, PS = dde.NeumannBC, dde.DirichletBC, dde.PointSetBC
    bcs = [DB(geom, lambda x_: sig(x_[:, 1:2]), on_i),
           NB(geom, lambda x_: 0.0, on_r),
           PS(pts.astype(np.float32), yobs)]
    data = dde.data.PDE(geom, pde, bcs, num_domain=4000, num_boundary=500)
    try:
        net = dde.nn.FNN([2, 64, 64, 64, 1], "tanh", "Glorot uniform")
    except AttributeError:
        net = dde.maps.FNN([2, 64, 64, 64, 1], "tanh", "Glorot uniform")
    try:
        sc = np.array([1.0 / Ls, 1.0])
        net.apply_feature_transform(lambda v: v * sc)
    except Exception:
        pass
    m = dde.Model(data, net)
    m.compile("adam", lr=1e-3)
    m.train(iterations=ITER, display_every=ITER)
    for r_ in range(RESTARTS):                         # L-BFGS x2 (v2)
        try:
            m.compile("L-BFGS")
            m.train(iterations=LBFGS_ITER, display_every=LBFGS_ITER)
        except Exception:
            break
    p2 = np.stack([Xg.ravel(), Yg.ravel()], 1).astype(np.float32)
    Cq = m.predict(p2).reshape(len(y), len(x))
    l2 = float(np.linalg.norm(Cq - Cr) / np.linalg.norm(Cr))
    lm = L_mix_interp(x, Cq)
    rows.append(dict(sigma=sn, L2=100.0 * l2, L_mix=lm))
    print("   L2 = {:.2%} | L_mix* = {:.3f}".format(l2, lm))
    with open("bruit.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

# ═══════════════ FIGURE ═══════════════
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
s = [100 * r["sigma"] for r in rows]
ax[0].plot(s, [r["L2"] for r in rows], "o-")
ax[0].set_xlabel(r"noise level $\sigma_n$ (%)"); ax[0].set_ylabel(r"$L_2$ error (%)")
ax[1].plot(s, [r["L_mix"] for r in rows], "s-")
ax[1].axhline(rows[0]["L_mix"], ls="--", color="k", lw=1)
ax[1].set_xlabel(r"noise level $\sigma_n$ (%)"); ax[1].set_ylabel(r"$L^*_{\rm mix}$")
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("figure_bruit.png", dpi=300)
print(json.dumps(rows, indent=2))