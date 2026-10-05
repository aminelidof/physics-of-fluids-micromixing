# -*- coding: utf-8 -*-
"""
============================================================================
 run_micromix.py (v3 — CORRIGE)
 CORRECTIF DECISIF : dc_yy = dde.grad.hessian(y, x, i=1, j=1)
   (l'ancien i=0, j=1 calculait la derivee CROISEE d2c/dxdy, ce qui
    rendait sig(y) solution exacte de la mauvaise equation -> loss
    parfaite mais champ faux. Voir diagnostic.)
 Autres correctifs : tf.random.set_seed (seeds reellement differents).
 Figures en anglais, sans titres (pretes pour le manuscrit).
 5 seeds : moyenne +/- sigma. Duree ~ 70 min (Adam 10000 + L-BFGS).
============================================================================
"""
import os, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("DDE_BACKEND", "tensorflow")
import deepxde as dde

# ═══════════════════ PARAMETRES ═══════════════════
L, H  = 5.0e-4, 1.0e-4     # canal [m] -> Ls = 5 (domaine utile)
D     = 1.0e-9             # diffusivite moleculaire [m2/s]
U     = 1.0e-4             # vitesse debitante [m/s]
KAPPA = 8.0                # principe TRIZ 24 : agitation magnetique
T_END = 10.0               # Ts = 1 (>= 2 transits pour la FDM)
DELTA = 0.1                # epaisseur d'interface (adoucie)

ITER, LBFGS_ITER = 10000, 2000
LR, DISPLAY_EVERY = 1e-3, 100
SEEDS = [42, 1, 7, 123, 2024]
WIDTH, DEPTH = 64, 4

PE = U * H / D
Ls, Ts = L / H, D * T_END / (H * H)
RUN_DIR = "results_" + datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(RUN_DIR, exist_ok=True)

def save_json(name, obj):
    with open(os.path.join(RUN_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

save_json("config.json", dict(L=L, H=H, D=D, U=U, KAPPA=KAPPA, T_END=T_END,
          DELTA=DELTA, PE=PE, Ls=Ls, Ts=Ts, ITER=ITER, LBFGS_ITER=LBFGS_ITER,
          LR=LR, DISPLAY_EVERY=DISPLAY_EVERY, SEEDS=SEEDS, WIDTH=WIDTH,
          DEPTH=DEPTH, formulation="steady PINN vs time-marching FDM",
          fix="hessian i=1 j=1 pour d2c/dy2"))
print("Dossier :", RUN_DIR, "| Pe = {:.2f} | Ls = {:.1f}".format(PE, Ls))

# ═══════════════════ FONCTIONS COMMUNES ═══════════════════
def sig(x, delta=DELTA):
    return 1.0 / (1.0 + np.exp(-(np.asarray(x) - 0.5) / delta))

def reference_fdm(kappa, nx=250, ny=25):
    """FDM time-marching (upwind) -> champ de regime etabli a t = Ts."""
    dx, dy = Ls / (nx - 1), 1.0 / (ny - 1)
    dt = min(0.2 * dy * dy / kappa, 0.2 * dx * dx, 0.4 * dx / (PE * 1.5))
    nt = max(50, int(np.ceil(Ts / dt)))
    dt = Ts / nt
    rx, ry = dt / (dx * dx), kappa * dt / (dy * dy)
    x = np.linspace(0.0, Ls, nx); y = np.linspace(0.0, 1.0, ny)
    u = 6.0 * y * (1.0 - y)
    X, Y = np.meshgrid(x, y)                      # (ny, nx)
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

def mixing_length(x, C, frac=0.05):
    var_y = C.var(axis=0)
    idx = np.where(var_y <= frac * var_y[0])[0]
    return (float(x[idx[0]]), var_y) if len(idx) else (float(x[-1]), var_y)

def save_loss_history(steps, comps, seed):
    if not comps:
        return
    n = len(comps[0])
    rows = np.array([[s] + list(c) + [float(np.sum(c))] for s, c in zip(steps, comps)])
    header = "step," + ",".join(["loss_pde"] + ["loss_bc{}".format(i) for i in range(1, n)]) + ",loss_total"
    np.savetxt(os.path.join(RUN_DIR, "loss_history_seed{}.csv".format(seed)),
               rows, delimiter=",", header=header, comments="")

def plot_convergence(steps, comps, seed):
    if not comps:
        return
    lt = np.asarray(comps, float); st = np.asarray(steps, float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(st, lt.sum(axis=1), lw=2, label="Total loss")
    if lt.shape[1] > 1:
        ax.semilogy(st, lt[:, 0], lw=1, alpha=.7, label="PDE residual")
        ax.semilogy(st, lt[:, 1:].sum(axis=1), lw=1, alpha=.7, label="Boundary terms")
    ax.set_xlabel("Iteration (Adam + L-BFGS)")
    ax.set_ylabel("Loss (log scale)")
    ax.grid(alpha=.3, which="both"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RUN_DIR, "convergence_seed{}.png".format(seed)), dpi=300)
    plt.close(fig)

# ═══════════════ PINN STATIONNAIRE (regime etabli) ═══════════════
def build_and_train(seed):
    """Pe*u*(y) dc/dx = d2c/dx2 + kappa*d2c/dy2 — CORRIGE."""
    np.random.seed(seed)
    try:
        dde.config.set_random_seed(seed)
    except Exception:
        pass
    try:                                    # fix : diversifier reellement les seeds
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
    geom = dde.geometry.Rectangle([0.0, 0.0], [Ls, 1.0])

    def pde(x, y):
        dc_x  = dde.grad.jacobian(y, x, i=0, j=0)      # dc/dx
        dc_xx = dde.grad.hessian(y, x, i=0, j=0)       # d2c/dx2
        dc_yy = dde.grad.hessian(y, x, i=1, j=1)       # d2c/dy2  <-- CORRECTIF
        u_val = 6.0 * x[:, 1:2] * (1.0 - x[:, 1:2])    # Poiseuille
        return PE * u_val * dc_x - dc_xx - KAPPA * dc_yy

    def on_inlet(x, ob): return ob and np.isclose(x[0], 0.0)
    def on_rest(x, ob):  return ob and not np.isclose(x[0], 0.0)

    try:
        NBcls, DirBC = dde.icbc.NeumannBC, dde.icbc.DirichletBC
    except AttributeError:
        NBcls, DirBC = dde.NeumannBC, dde.DirichletBC
    bcs = [DirBC(geom, lambda x: sig(x[:, 1:2]), on_inlet),
           NBcls(geom, lambda x: 0.0, on_rest)]

    data = dde.data.PDE(geom, pde, bcs, num_domain=8000, num_boundary=1000)
    dims = [2] + [WIDTH] * DEPTH + [1]
    try:
        net = dde.nn.FNN(dims, "tanh", "Glorot uniform")
    except AttributeError:
        net = dde.maps.FNN(dims, "tanh", "Glorot uniform")
    try:                                    # normalisation des entrees (anti-saturation)
        scale = np.array([1.0 / Ls, 1.0])
        net.apply_feature_transform(lambda x: x * scale)
    except Exception:
        pass

    model = dde.Model(data, net)
    model.compile("adam", lr=LR)
    lh1, ts1 = model.train(iterations=ITER, display_every=DISPLAY_EVERY)
    steps = list(lh1.steps); comps = list(lh1.loss_train)
    try:                                    # raffinement L-BFGS
        model.compile("L-BFGS")
        lh2, ts2 = model.train(iterations=LBFGS_ITER, display_every=LBFGS_ITER)
        if lh2.steps and lh2.loss_train:
            off = (steps[-1] if steps else 0)
            steps += [off + s for s in lh2.steps]
            comps += list(lh2.loss_train)
    except Exception as e:
        print("   (L-BFGS ignore : {})".format(e))
    best = float(np.min([np.sum(c) for c in comps])) if comps else float("nan")
    return model, steps, comps, best

def predict_field(model, x, y):
    Xg, Yg = np.meshgrid(x, y)
    pts = np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32)
    return model.predict(pts).reshape(len(y), len(x))

# ═══════════════════ EXECUTION ═══════════════════
t0 = time.time()
print("[1/5] References FDM (time-marching -> regime etabli)...")
x, y, C_ref_before = reference_fdm(1.0)
_, _, C_ref_after = reference_fdm(KAPPA)
L_b, var_b     = mixing_length(x, C_ref_before)
L_ref, var_ref = mixing_length(x, C_ref_after)
print("   FDM : L_mix* avant = {:.3f} | apres = {:.3f}".format(L_b, L_ref))

print("[2/5] PINN stationnaire : {} seeds (Adam {} + L-BFGS {})...".format(
      len(SEEDS), ITER, LBFGS_ITER))
stats = []
for k, seed in enumerate(SEEDS, 1):
    ts_ = time.time()
    model, steps, comps, best = build_and_train(seed)
    save_loss_history(steps, comps, seed)
    plot_convergence(steps, comps, seed)
    C_pinn = predict_field(model, x, y)
    l2 = float(np.linalg.norm(C_pinn - C_ref_after) / np.linalg.norm(C_ref_after))
    L_p, var_p = mixing_length(x, C_pinn)
    stats.append(dict(seed=seed, l2=l2, L_mix=L_p, loss=best))
    flag = "  *** ATTENTION : L2 > 10% — non converge ***" if l2 > 0.10 else ""
    print("   [{}/{}] seed {} : L2 = {:.2%} | L_mix* = {:.3f} | loss = {:.2e} | {:.0f} s{}".format(
          k, len(SEEDS), seed, l2, L_p, best, time.time() - ts_, flag))
    last = dict(C_pinn=C_pinn, var_p=var_p, L_p=L_p)

C_pinn, var_p, L_p = last["C_pinn"], last["var_p"], last["L_p"]

print("[3/5] Metriques (moyennes {} seeds)...".format(len(SEEDS)))
m_b = L_b * H / U
l2s     = [s["l2"] for s in stats]
gains   = [100.0 * (m_b - s["L_mix"] * H / U) / m_b for s in stats]
lm_list = [s["L_mix"] for s in stats]
lm_mean, lm_std = float(np.mean(lm_list)), float(np.std(lm_list))
gain_mean, gain_std = float(np.mean(gains)), float(np.std(gains))
l2_mean,  l2_std    = float(np.mean(l2s)),  float(np.std(l2s))
m_a = lm_mean * H / U
gain, gain_theo = gain_mean, 100.0 * (1.0 - 1.0 / KAPPA)

print("[4/5] Figures et exports (anglais, sans titres)...")
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
im0 = ax[0].imshow(C_ref_before, origin="lower", extent=[0, Ls, 0, 1],
                   aspect="auto", cmap="viridis", vmin=0, vmax=1)
im1 = ax[1].imshow(C_pinn, origin="lower", extent=[0, Ls, 0, 1],
                   aspect="auto", cmap="viridis", vmin=0, vmax=1)
fig.colorbar(im0, ax=ax[0], label="c")
fig.colorbar(im1, ax=ax[1], label="c")
for a in (ax[0], ax[1]):
    a.set_xlabel("x*"); a.set_ylabel("y*")
ax[2].semilogy(x, var_b / var_b[0], label="Baseline ($\\kappa$ = 1)")
ax[2].semilogy(x, var_p / var_b[0], label="Magnet-assisted (PINN, $\\kappa$ = {:.0f})".format(KAPPA))
ax[2].axhline(0.05, ls="--", color="k", lw=1, label="5% threshold")
ax[2].axvline(L_b, color="tab:blue", ls=":")
ax[2].axvline(L_p, color="tab:orange", ls=":")
ax[2].set_xlabel("x*"); ax[2].set_ylabel("Normalized sectional variance")
ax[2].grid(alpha=.3, which="both"); ax[2].legend()
fig.tight_layout()
fig.savefig(os.path.join(RUN_DIR, "figure1_champs.png"), dpi=300)
plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for xs in (0.5, 1.5, 4.0):
    j = int(np.argmin(np.abs(x - xs)))
    ax[0].plot(y, C_ref_after[:, j], "o", ms=4, label="FDM, x* = {:.1f}".format(xs))
    ax[0].plot(y, C_pinn[:, j], "-", lw=1.6, label="PINN, x* = {:.1f}".format(xs))
ax[0].set_xlabel("y*"); ax[0].set_ylabel("c(y*)")
ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
err = np.abs(C_pinn - C_ref_after)
im = ax[1].imshow(err, origin="lower", extent=[0, Ls, 0, 1], aspect="auto", cmap="magma")
fig.colorbar(im, ax=ax[1], label="|PINN - FDM|")
ax[1].set_xlabel("x*"); ax[1].set_ylabel("y*")
fig.tight_layout()
fig.savefig(os.path.join(RUN_DIR, "figure2_validation.png"), dpi=300)
plt.close(fig)

Xg, Yg = np.meshgrid(x, y)
np.savetxt(os.path.join(RUN_DIR, "champs.csv"),
           np.column_stack([Xg.ravel(), Yg.ravel(), C_ref_before.ravel(),
                            C_ref_after.ravel(), C_pinn.ravel()]),
           delimiter=",", header="x_star,y_star,c_avant,c_apres_ref,c_apres_pinn", comments="")
np.savez_compressed(os.path.join(RUN_DIR, "champs.npz"), x=x, y=y,
                    c_avant=C_ref_before, c_apres_ref=C_ref_after, c_apres_pinn=C_pinn)
np.savetxt(os.path.join(RUN_DIR, "variance_profils.csv"),
           np.column_stack([x, var_b / var_b[0], var_ref / var_b[0], var_p / var_b[0]]),
           delimiter=",", header="x_star,var_avant,var_apres_ref,var_apres_pinn", comments="")

save_json("metriques.json", dict(
    Pe=PE, Ls=Ls, Ts=Ts, n_seeds=len(SEEDS), seeds=SEEDS,
    L_mix_avant_x=L_b, L_mix_avant_m=L_b * H, t_mix_avant_s=m_b,
    L_mix_apres_pinn_x=lm_mean, L_mix_apres_pinn_m=lm_mean * H, t_mix_apres_s=m_a,
    L_mix_apres_ref_x=L_ref, figures="dernier seed (representatif)",
    gain_pct=gain, gain_theorique_pct=gain_theo,
    erreur_L2_pct=100.0 * l2_mean,
    loss_best=float(np.nan_to_num(stats[-1]["loss"])),
    stats_seeds=dict(n=len(SEEDS), L2_mean=l2_mean, L2_std=l2_std,
                     gain_mean=gain_mean, gain_std=gain_std,
                     L_mix_mean=lm_mean, L_mix_std=lm_std),
    par_seed=[dict(seed=s["seed"], l2=s["l2"], L_mix=s["L_mix"], loss=s["loss"]) for s in stats]))

duree = time.time() - t0
snippet = ("A physics-informed neural network (DeepXDE) solves the steady 2D "
           "advection-dispersion equation with a Poiseuille profile (Pe = {:.1f}), "
           "while an independent time-marching finite-difference solver provides the "
           "cross-validation (relative L2 error of {:.2f} +/- {:.2f}% over {} runs). "
           "The magnetically intensified dispersion (kappa = {:.0f}, TRIZ principle 24) "
           "reduces the mixing length from {:.2f} mm to {:.2f} mm, i.e. the mixing time "
           "from {:.2f} s to {:.2f} s (gain of {:.1f} +/- {:.1f}%), without any increase "
           "in driving pressure.").format(PE, 100.0 * l2_mean, 100.0 * l2_std,
           len(SEEDS), KAPPA, L_b * H * 1000, lm_mean * H * 1000, m_b, m_a, gain_mean, gain_std)

save_json("resume.json", dict(date=datetime.now().isoformat(), duree_s=round(duree, 1),
          dossier=RUN_DIR, parametres=dict(L=L, H=H, D=D, U=U, KAPPA=KAPPA, T_END=T_END),
          resultats=dict(gain_pct=round(gain_mean, 1), gain_std=round(gain_std, 1),
                         erreur_L2_pct=round(100.0 * l2_mean, 2),
                         t_mix_avant_s=round(m_b, 2), t_mix_apres_s=round(m_a, 2)),
          article_snippet_en=snippet))

print("[5/5] TERMINE en {:.1f} s".format(duree))
print("=" * 70)
print("RESUME POUR L'ARTICLE  (controle qualite : L2 doit etre < 5%)")
print("  Pe = {:.2f} | {} seeds | L2 = {:.2%} +/- {:.2%}".format(
      PE, len(SEEDS), l2_mean, l2_std))
print("  gain = {:.1f} +/- {:.1f}% (theorie : {:.1f}%)".format(
      gain_mean, gain_std, gain_theo))
print("  L_mix : {:.3f} -> {:.3f} (x*)  |  t_mix : {:.2f} s -> {:.2f} s".format(
      L_b, lm_mean, m_b, m_a))
print("  Resultats : {}".format(RUN_DIR))
print("=" * 70)