# -*- coding: utf-8 -*-
"""
============================================================================
 run_verification.py (v2 — complet)
 Verification (critere 4) + metriques de validation (critere 5).
 Corrections integrees :
  - L_mix INTERPOLE au croisement du seuil (independance grille reelle) ;
  - R2 global + R2 de la zone de melange (x* <= 1) ou la variance vit ;
  - hessian i=1, j=1 ; feature transform ; tf seed.
 Duree : ~15 min (dont ~8 min pour l'etude de collocation).
 Sorties : verification.json + figure_verification.png
============================================================================
"""
import os, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("DDE_BACKEND", "tensorflow")

# ═══════════════ PARAMETRES (coherents run_micromix v3) ═══════════════
L, H, D, U, KAPPA = 5.0e-4, 1.0e-4, 1.0e-9, 1.0e-4, 8.0
Ls, DELTA, PE, Ts = L / H, 0.1, U * H / D, 1.0
GRID_N  = [125, 250, 500]          # etude de maillage FDM
COL_N   = [2000, 4000, 8000]       # etude de collocation PINN
COL_ITER, COL_LBFGS = 3000, 1000   # entrainements courts

def sig(x):
    return 1.0 / (1.0 + np.exp(-(np.asarray(x) - 0.5) / DELTA))

def solve_fdm(kappa, nx, ny, Ts=Ts):
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
    """Croisement du seuil INTERPOLE (independance grille reelle)."""
    v = C.var(axis=0); thr = frac * v[0]
    idx = np.where(v <= thr)[0]
    if not len(idx):
        return float(x[-1])
    k = idx[0]
    if k == 0:
        return float(x[0])
    v1, v2 = v[k - 1], v[k]
    return float(x[k] if v1 == v2 else
                 x[k - 1] + (v1 - thr) / (v1 - v2) * (x[k] - x[k - 1]))

# ═══════════════ 1. CHARGEMENT DU RUN VALID ═══════════════
d = sorted(glob.glob("results_[0-9]*"))[-1]
z = np.load(os.path.join(d, "champs.npz"))
x, y = z["x"], z["y"]
Cp, Cr = z["c_apres_pinn"], z["c_apres_ref"]
print("Source : {} (champs.npz)".format(d))

# ═══════════════ 2. METRIQUES DE VALIDATION (critere 5) ═══════════════
diff = (Cp - Cr).ravel()
l2   = float(np.linalg.norm(Cp - Cr) / np.linalg.norm(Cr))
rmse = float(np.sqrt(np.mean(diff ** 2)))
mae  = float(np.mean(np.abs(diff)))
sst  = float(np.sum((Cr - Cr.mean()) ** 2))
r2   = 1.0 - float(np.sum(diff ** 2)) / sst
# R2 restreint a la zone de melange (x* <= 1) : la seule ou la variance vit
jz = int(np.argmin(np.abs(x - 1.0)))
Cpz, Crz = Cp[:, :jz], Cr[:, :jz]
r2_zone = 1.0 - float(np.sum((Cpz - Crz) ** 2)) / \
                float(np.sum((Crz - Crz.mean()) ** 2))
print("L2 = {:.2%} | RMSE = {:.1e} | MAE = {:.1e} | R2 = {:.4f} | R2(zone) = {:.4f}".format(
      l2, rmse, mae, r2, r2_zone))

# ═══════════════ 3. CONSERVATION (critere 4) ═══════════════
cons_p = abs(float(Cp.mean()) - 0.5)
cons_f = abs(float(Cr.mean()) - 0.5)
print("Conservation |<c>-0.5| : PINN = {:.1e} | FDM = {:.1e}".format(cons_p, cons_f))

# ═══════════════ 4. RESIDUS (depuis loss_history) ═══════════════
lf = sorted(glob.glob(os.path.join(d, "loss_history_seed*.csv")))
res_fin, bc_fin = float("nan"), float("nan")
if lf:
    hist = np.genfromtxt(lf[-1], delimiter=",", names=True)
    res_fin = float(hist["loss_pde"][-1])
    bcs = [n for n in hist.dtype.names if n.startswith("loss_bc")]
    if bcs:
        bc_fin = float(sum(hist[n][-1] for n in bcs))
print("Residu PDE final = {:.1e} | loss BC finales = {:.1e}".format(res_fin, bc_fin))

# ═══════════════ 5. INDEPENDANCE DE MAILLAGE (FDM) ═══════════════
Lm = []
for nx in GRID_N:
    ny = max(13, int(0.1 * nx))
    xg, yg, Cg = solve_fdm(KAPPA, nx, ny)
    Lm.append(L_mix_interp(xg, Cg))
    print("Grille {:4d}x{:3d} : L_mix* = {:.4f}".format(nx, ny, Lm[-1]))

# ═══════════════ 6. INDEPENDANCE DE COLLOCATION (PINN courts) ═══════════════
col_l2 = []
for nf in COL_N:
    print("Collocation N_f = {} (entrainement court)...".format(nf))
    try:
        import deepxde as dde
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
            dc_yy = dde.grad.hessian(y_, x_, i=1, j=1)   # CORRECTIF (i=1, j=1)
            uv = 6.0 * x_[:, 1:2] * (1.0 - x_[:, 1:2])
            return PE * uv * dc_x - dc_xx - KAPPA * dc_yy
        on_i = lambda x_, ob: ob and np.isclose(x_[0], 0.0)
        on_r = lambda x_, ob: ob and not np.isclose(x_[0], 0.0)
        try:
            NB, DB = dde.icbc.NeumannBC, dde.icbc.DirichletBC
        except AttributeError:
            NB, DB = dde.NeumannBC, dde.DirichletBC
        bcs = [DB(geom, lambda x_: sig(x_[:, 1:2]), on_i),
               NB(geom, lambda x_: 0.0, on_r)]
        data = dde.data.PDE(geom, pde, bcs, num_domain=nf,
                            num_boundary=max(200, nf // 8))
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
        m.train(iterations=COL_ITER, display_every=COL_ITER)
        try:
            m.compile("L-BFGS")
            m.train(iterations=COL_LBFGS, display_every=COL_LBFGS)
        except Exception:
            pass
        Xg, Yg = np.meshgrid(x, y)
        pts = np.stack([Xg.ravel(), Yg.ravel()], 1).astype(np.float32)
        Cq = m.predict(pts).reshape(len(y), len(x))
        col_l2.append(float(np.linalg.norm(Cq - Cr) / np.linalg.norm(Cr)))
    except Exception as e:
        print("   (skip : {})".format(e))
        col_l2.append(float("nan"))
    print("   L2 = {:.2%}".format(col_l2[-1]))

# ═══════════════ 7. SAUVEGARDES ═══════════════
res = dict(source=d, L2_pct=100.0 * l2, RMSE=rmse, MAE=mae,
           R2=r2, R2_zone=r2_zone,
           conservation_pin=cons_p, conservation_fdm=cons_f,
           residu_pde=res_fin, bc_fin=bc_fin,
           grid_N=GRID_N, L_mix_grids=Lm,
           col_N=COL_N, L2_col=col_l2)
with open("verification.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2)

fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.4))
if lf:
    ax[0].semilogy(hist["step"], hist["loss_pde"], lw=1.2, label="PDE residual")
    ax[0].semilogy(hist["step"], hist["loss_total"], lw=1.8, label="Total loss")
    ax[0].legend(fontsize=8)
ax[0].set_xlabel("Iteration"); ax[0].set_ylabel("Loss (log scale)")
ax[1].bar(["PINN", "FDM"], [cons_p, cons_f], color=["tab:blue", "tab:gray"])
ax[1].set_ylabel(r"$|\langle c \rangle - 0.5|$")
ax[2].plot(GRID_N, Lm, "o-")
ax[2].set_xlabel(r"FDM grid $n_x$"); ax[2].set_ylabel(r"$L^*_{\rm mix}$ (interpolated)")
ax[3].plot(COL_N, col_l2, "s-")
ax[3].set_xlabel(r"$N_f$"); ax[3].set_ylabel(r"$L_2$ error")
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figure_verification.png", dpi=300)
print(json.dumps(res, indent=2))