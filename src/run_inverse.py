# -*- coding: utf-8 -*-
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("DDE_BACKEND", "tensorflow")
import deepxde as dde

L, H, D, U, KAPPA_TRUE = 5.0e-4, 1.0e-4, 1.0e-9, 1.0e-4, 8.0
Ls, DELTA, PE, Ts = L / H, 0.1, U * H / D, 1.0
NOBS, ITER, LBFGS_ITER, RESTARTS = 400, 6000, 3000, 2
X_OBS_MAX = 1.5                       # capteurs dans la zone informative
NOISE = [0.0, 0.01, 0.05]

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

def get_kappa(var):
    """Valeur de dde.Variable — gere propriete OU methode (selon version)."""
    kv = var.value
    if callable(kv):
        kv = kv()
    return float(np.asarray(kv).ravel()[0])

x, y, Ctrue = solve_fdm(KAPPA_TRUE)
rng = np.random.default_rng(7)
pts = np.column_stack([rng.uniform(0, X_OBS_MAX, NOBS),
                       rng.uniform(0, 1, NOBS)])
from numpy import searchsorted
i0 = np.clip(searchsorted(x, pts[:, 0]) - 1, 0, len(x) - 2)
j0 = np.clip(searchsorted(y, pts[:, 1]) - 1, 0, len(y) - 2)
tx = (pts[:, 0] - x[i0]) / (x[i0 + 1] - x[i0])
ty = (pts[:, 1] - y[j0]) / (y[j0 + 1] - y[j0])
cobs = (Ctrue[j0, i0] * (1 - tx) * (1 - ty) + Ctrue[j0, i0 + 1] * tx * (1 - ty)
        + Ctrue[j0 + 1, i0] * (1 - tx) * ty + Ctrue[j0 + 1, i0 + 1] * tx * ty)
print("Observations : {} points dans x* <= {} (zone de melange)".format(NOBS, X_OBS_MAX))

rows, curves = [], []
for sn in NOISE:
    print("=" * 60)
    print("Inverse, bruit = {:.0%}".format(sn))
    try:
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
        kappa = dde.Variable(1.0)              # inconnue, init = 1 (aucun prior)
        geom = dde.geometry.Rectangle([0.0, 0.0], [Ls, 1.0])

        def pde(x_, y_):
            dc_x  = dde.grad.jacobian(y_, x_, i=0, j=0)
            dc_xx = dde.grad.hessian(y_, x_, i=0, j=0)
            dc_yy = dde.grad.hessian(y_, x_, i=1, j=1)   # CORRECTIF (i=1, j=1)
            uv = 6.0 * x_[:, 1:2] * (1.0 - x_[:, 1:2])
            return PE * uv * dc_x - dc_xx - kappa * dc_yy

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
        m.compile("adam", lr=1e-3, external_trainable_variables=[kappa])
        m.train(iterations=ITER, display_every=ITER)
        k_adam = get_kappa(kappa)
        print("   apres Adam : kappa_hat = {:.3f}".format(k_adam))

        # --- Raffinements L-BFGS (restarts) : decisifs pour kappa ---
        for r_ in range(RESTARTS):
            try:
                m.compile("L-BFGS", external_trainable_variables=[kappa])
                m.train(iterations=LBFGS_ITER, display_every=LBFGS_ITER)
            except Exception as e:
                print("   (L-BFGS #{} ignore : {})".format(r_ + 1, e))
                break
            print("   apres L-BFGS #{} : kappa_hat = {:.3f}".format(
                  r_ + 1, get_kappa(kappa)))

        khat = get_kappa(kappa)
        err = 100.0 * abs(khat - KAPPA_TRUE) / KAPPA_TRUE
        print("   FINAL : kappa_hat = {:.3f} (vrai {:.1f}, erreur {:.2f}%)".format(
              khat, KAPPA_TRUE, err))
        rows.append(dict(sigma=sn, k_adam=k_adam, khat=khat, err=err))
        curves.append((sn, k_adam, khat))
    except Exception as e:
        print("   ERREUR : {}".format(e))
    with open("inverse.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

fig, ax = plt.subplots(figsize=(7, 4.2))
if rows:
    pos = np.arange(len(rows))
    ax.bar(pos - 0.18, [r["k_adam"] for r in rows], width=0.36,
           color="tab:orange", label="after Adam")
    ax.bar(pos + 0.18, [r["khat"] for r in rows], width=0.36,
           color="tab:blue", label="after L-BFGS")
    ax.axhline(KAPPA_TRUE, ls="--", color="k", label=r"$\kappa_{\rm true} = 8$")
    for i, r in enumerate(rows):
        ax.text(i + 0.18, r["khat"] + 0.12, "{:.2f}".format(r["khat"]), ha="center")
    ax.set_xticks(pos)
    ax.set_xticklabels(["{:.0%}".format(r["sigma"]) for r in rows])
    ax.set_xlabel(r"noise $\sigma_n$")
    ax.set_ylabel(r"$\hat{\kappa}$")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
fig.tight_layout()
fig.savefig("figure_inverse.png", dpi=300)
print(json.dumps(rows, indent=2))
