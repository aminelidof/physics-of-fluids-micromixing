# -*- coding: utf-8 -*-

import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

L, H, D = 5.0e-4, 1.0e-4, 1.0e-9     
Ls = L / H
DELTA = 0.1                            
NX, NY, FRAC = 250, 25, 0.05

PE_LIST    = [2.5, 5.0, 10.0, 20.0]
KAPPA_LIST = [1.0, 2.0, 4.0, 8.0]     

PINN_CHECK = [(10.0, 8.0), (5.0, 4.0)] 
PINN_ITER  = 8000

RUN_DIR = "results_parametrique_" + datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(RUN_DIR, exist_ok=True)

def save_json(name, obj):
    with open(os.path.join(RUN_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def sig(x, delta=DELTA):
    return 1.0 / (1.0 + np.exp(-(np.asarray(x) - 0.5) / delta))

def solve_fdm(PE, kappa, Ts, nx=NX, ny=NY):
    dx, dy = Ls / (nx - 1), 1.0 / (ny - 1)
    dt = min(0.2 * dy * dy / kappa, 0.2 * dx * dx, 0.4 * dx / (PE * 1.5))
    nt = max(50, int(np.ceil(Ts / dt)))
    dt = Ts / nt
    rx, ry = dt / (dx * dx), kappa * dt / (dy * dy)
    x = np.linspace(0.0, Ls, nx); y = np.linspace(0.0, 1.0, ny)
    u = 6.0 * y * (1.0 - y)
    X, Y = np.meshgrid(x, y)
    C = sig(Y); inlet = sig(y)
    for n in range(nt):
        Cn = C.copy()
        core = Cn[1:-1, 1:-1]
        adv = dt * PE * u[1:-1, None] * (core - Cn[1:-1, :-2]) / dx
        C[1:-1, 1:-1] = core - adv \
            + rx * (Cn[1:-1, 2:] - 2.0 * core + Cn[1:-1, :-2]) \
            + ry * (Cn[2:, 1:-1] - 2.0 * core + Cn[:-2, 1:-1])
        C[:, 0] = inlet; C[:, -1] = C[:, -2]
        C[0, :] = C[1, :]; C[-1, :] = C[-2, :]
    return x, y, C

def mixing_length(x, C, frac=FRAC):
    var_y = C.var(axis=0)
    idx = np.where(var_y <= frac * var_y[0])[0]
    return float(x[idx[0]]) if len(idx) else float(x[-1])

def t_mix(Lstar, PE):
    return Lstar * H * H / (PE * D)

def pinn_L_mix(PE, kappa, iterations):
    os.environ.setdefault("DDE_BACKEND", "tensorflow")
    import deepxde as dde
    try:
        dde.config.set_random_seed(42)
    except Exception:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(42)
    except Exception:
        pass
    geom = dde.geometry.Rectangle([0.0, 0.0], [Ls, 1.0])
    def pde(x, y):
        dc_x  = dde.grad.jacobian(y, x, i=0, j=0)      
        dc_xx = dde.grad.hessian(y, x, i=0, j=0)       
        dc_yy = dde.grad.hessian(y, x, i=1, j=1)      
        u_val = 6.0 * x[:, 1:2] * (1.0 - x[:, 1:2])    
        return PE * u_val * dc_x - dc_xx - kappa * dc_yy
    def on_inlet(x, ob): return ob and np.isclose(x[0], 0.0)
    def on_rest(x, ob):  return ob and not np.isclose(x[0], 0.0)
    try:
        NBcls, DirBC = dde.icbc.NeumannBC, dde.icbc.DirichletBC
    except AttributeError:
        NBcls, DirBC = dde.NeumannBC, dde.DirichletBC
    bcs = [DirBC(geom, lambda x: sig(x[:, 1:2]), on_inlet),
           NBcls(geom, lambda x: 0.0, on_rest)]
    data = dde.data.PDE(geom, pde, bcs, num_domain=8000, num_boundary=1000)
    try:
        net = dde.nn.FNN([2, 64, 64, 64, 1], "tanh", "Glorot uniform")
    except AttributeError:
        net = dde.maps.FNN([2, 64, 64, 64, 1], "tanh", "Glorot uniform")
    try:
        scale = np.array([1.0 / Ls, 1.0])
        net.apply_feature_transform(lambda x: x * scale)
    except Exception:
        pass
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    model.train(iterations=iterations, display_every=iterations)
    try:
        model.compile("L-BFGS")
        model.train(iterations=2000, display_every=2000)
    except Exception:
        pass
    Ts = max(2.0 * Ls / PE, 0.5)          
    x, y, C_ref = solve_fdm(PE, kappa, Ts)
    Xg, Yg = np.meshgrid(x, y)
    pts = np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32)
    C_p = model.predict(pts).reshape(len(y), len(x))
    l2 = float(np.linalg.norm(C_p - C_ref) / np.linalg.norm(C_ref))
    return mixing_length(x, C_p), l2

t0 = time.time()
rows, pinn_rows = [], []
GAIN = np.zeros((len(KAPPA_LIST), len(PE_LIST)))
LMIX = np.zeros_like(GAIN)

for j, pe in enumerate(PE_LIST):
    Ts = max(2.0 * Ls / pe, 0.5)
    x, y, C1 = solve_fdm(pe, 1.0, Ts)
    L1 = mixing_length(x, C1)
    for i, ka in enumerate(KAPPA_LIST):
        x, y, Ck = solve_fdm(pe, ka, Ts)
        Lk = mixing_length(x, Ck)
        g = 100.0 * (1.0 - Lk / L1)
        GAIN[i, j], LMIX[i, j] = g, Lk
        rows.append([pe, ka, Ts, L1, Lk, Lk * H, t_mix(Lk, pe), g])
print("Balayage FDM termine en {:.1f} s".format(time.time() - t0))

for pe_c, ka_c in PINN_CHECK:
    print("Validation PINN (Pe={:.1f}, kappa={:.1f})...".format(pe_c, ka_c))
    L_pinn, l2 = pinn_L_mix(pe_c, ka_c, PINN_ITER)
    Ts = max(2.0 * Ls / pe_c, 0.5)
    xg, yg, Ck = solve_fdm(pe_c, ka_c, Ts)
    L_fdm = mixing_length(xg, Ck)
    pinn_rows.append([pe_c, ka_c, L_pinn, L_fdm,
                      abs(L_pinn - L_fdm) / L_fdm * 100.0, l2 * 100.0])
    print("   L_mix* PINN = {:.3f} | FDM = {:.3f} | L2 = {:.2%}".format(L_pinn, L_fdm, l2))

def cell_edges(v):
    v = np.asarray(v, float)
    mids = (v[:-1] + v[1:]) / 2.0
    return np.concatenate([[2 * v[0] - mids[0]], mids, [2 * v[-1] - mids[-1]]])

fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6))
xe, ke = cell_edges(PE_LIST), cell_edges(KAPPA_LIST)
pc0 = ax[0].pcolormesh(xe, ke, GAIN, cmap="RdYlGn", vmin=0.0, vmax=100.0, shading="flat")
pc1 = ax[1].pcolormesh(xe, ke, LMIX, cmap="viridis", shading="flat")
for a, pc, clabel in ((ax[0], pc0, "Mixing gain (%)"),
                      (ax[1], pc1, "L$_{\\mathrm{mix}}^{*}$")):
    a.set_yscale("log")
    a.set_yticks(KAPPA_LIST)
    a.set_yticklabels(["{:.0f}".format(k) for k in KAPPA_LIST])
    a.set_xticks(PE_LIST)
    a.set_xticklabels(["{:.1f}".format(p) for p in PE_LIST])
    a.set_xlabel("Pe = UH/D")
    a.set_ylabel("$\\kappa$ (magnetic agitation)")
    fig.colorbar(pc, ax=a, label=clabel)
for i in range(len(KAPPA_LIST)):
    for j in range(len(PE_LIST)):
        ax[0].text(PE_LIST[j], KAPPA_LIST[i], "{:.0f}%".format(GAIN[i, j]),
                   ha="center", va="center", fontsize=9, fontweight="bold")
for pe_c, ka_c in PINN_CHECK:
    ax[0].plot(pe_c, ka_c, "*", ms=16, mec="k", mfc="white", mew=1.2)
    ax[1].plot(pe_c, ka_c, "*", ms=16, mec="k", mfc="white", mew=1.2)
fig.tight_layout()
fig.savefig(os.path.join(RUN_DIR, "carte_gain.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

np.savetxt(os.path.join(RUN_DIR, "parametrique.csv"), np.array(rows), delimiter=",",
           header="Pe,kappa,Ts,L_mix_kappa1,L_mix,L_mix_m,t_mix_s,gain_pct", comments="")
if pinn_rows:
    np.savetxt(os.path.join(RUN_DIR, "validation_pinn.csv"), np.array(pinn_rows), delimiter=",",
               header="Pe,kappa,L_mix_pinn,L_mix_fdm,ecart_L_pct,erreur_L2_pct", comments="")
save_json("resume.json", dict(
    date=datetime.now().isoformat(), duree_s=round(time.time() - t0, 1),
    PE_LIST=PE_LIST, KAPPA_LIST=KAPPA_LIST, H=H, D=D, L=L,
    pinn_check=[list(p) for p in PINN_CHECK],
    interpretation=("A faible Pe, la diffusion longitudinale participe au melange "
                    "et le gain de l'agitation est amorti ; a Pe eleve, le gain tend "
                    "vers la borne theorique (1 - 1/kappa)."),
    article_snippet_en=("A parametric study over Pe in [{:.1f}, {:.0f}] and kappa in "
                        "[{:.0f}, {:.0f}] shows that the mixing gain ranges from {:.0f}% "
                        "to {:.0f}%, approaching the theoretical bound (1 - 1/kappa) "
                        "as Pe increases, while low-Pe regimes are partially homogenized "
                        "by longitudinal diffusion.").format(
                        PE_LIST[0], PE_LIST[-1], KAPPA_LIST[0], KAPPA_LIST[-1],
                        float(GAIN[1:, :].min()), float(GAIN.max()))))

print("Resultats sauvegardes dans : {}".format(RUN_DIR))
print("Gain min (kappa>1) = {:.0f}% | max = {:.0f}%".format(
      float(GAIN[1:, :].min()), float(GAIN.max())))
