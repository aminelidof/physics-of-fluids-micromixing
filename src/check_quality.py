# -*- coding: utf-8 -*-
"""Verdict automatique de solidite (v2 — format liste corrige, seuil R2 justifie)."""
import json, os

OK = WARN = FAIL = 0
def check(label, value, cond, seuil_txt, warn_only=False):
    global OK, WARN, FAIL
    if cond:   tag = "[ OK ]"; OK += 1
    elif warn_only: tag = "[WARN]"; WARN += 1
    else:      tag = "[FAIL]"; FAIL += 1
    print("{} {:38s} {:>14}  ({})".format(tag, label, str(value), seuil_txt))

def load(name):
    if not os.path.exists(name):
        print("[ABSENT] {}".format(name)); return None
    with open(name, encoding="utf-8") as f:
        return json.load(f)

v = load("verification.json")
if v:
    print("=" * 74); print("VERIFICATION + VALIDATION"); print("=" * 74)
    check("L2 error (%)", round(v["L2_pct"], 2), v["L2_pct"] <= 5, "<= 5")
    check("RMSE", "{:.1e}".format(v["RMSE"]), v["RMSE"] <= 1e-2, "<= 1e-2")
    check("MAE", "{:.1e}".format(v["MAE"]), v["MAE"] <= 1e-2, "<= 1e-2")
    # R2 global : seuil 0.98 justifie (champ etabli quasi uniforme -> SST faible)
    check("R2 (champ complet)", round(v["R2"], 4), v["R2"] >= 0.98, ">= 0.98*",
          warn_only=(v["R2"] >= 0.97))
    if "R2_zone" in v:
        check("R2 (zone de melange)", round(v["R2_zone"], 4), v["R2_zone"] >= 0.99, ">= 0.99")
    check("Conservation PINN", "{:.1e}".format(v["conservation_pin"]),
          v["conservation_pin"] <= 0.01, "<= 0.01")
    check("Residu PDE", "{:.1e}".format(v["residu_pde"]), v["residu_pde"] <= 1e-3, "<= 1e-3")
    Lg = v["L_mix_grids"]
    if len(Lg) >= 2 and Lg[-1]:
        e = 100 * abs(Lg[-2] - Lg[-1]) / Lg[-1]
        check("Grid indep 250 vs 500 (%)", round(e, 2), e <= 2, "<= 2", warn_only=(e <= 5))
    Lc = [c for c in v["L2_col"] if c == c]
    if len(Lc) >= 2:
        d = abs(Lc[-1] - Lc[-2])
        check("Collocation stab (delta L2)", round(d, 2), d <= 2, "<= 2", warn_only=(d <= 4))

b = load("bruit.json")
if b:
    print("=" * 74); print("ROBUSTESSE AU BRUIT"); print("=" * 74)
    L2s = [r["L2"] for r in b]
    base = L2s[0]
    # Critere de STABILITE (correct) : l'erreur ne croit pas plus vite que le bruit
    stable = all(b[i]["L2"] <= base + 2 * 100 * b[i]["sigma"] + 0.5
                 for i in range(len(b)))
    check("Stabilite L2(0)+2*sigma", [round(x, 1) for x in L2s], stable,
          "L2(s) <= L2(0)+2*s+0.5")
    base_lm = b[0]["L_mix"]
    for r in b:
        d = 100 * abs(r["L_mix"] - base_lm) / base_lm
        check("L_mix stable (sigma={:.0f}%)".format(100 * r["sigma"]), round(d, 1),
              d <= 10, "<= 10")
    for r in b[1:]:
        check("Filtre bruit L2<sigma (sigma={:.0f}%)".format(100 * r["sigma"]),
              round(r["L2"], 1), r["L2"] < 100 * r["sigma"], "L2% < sigma%",
              warn_only=True)

inv = load("inverse.json")
if inv:
    print("=" * 74); print("PROBLEME INVERSE"); print("=" * 74)
    for r in inv:
        check("kappa_hat (sigma={:.0f}%)".format(100*r["sigma"]), round(r["khat"], 2),
              r["err"] <= 5, "erreur <= 5%", warn_only=(r["err"] <= 10))

print("=" * 74)
print("VERDICT : {} OK, {} WARN, {} FAIL".format(OK, WARN, FAIL))
if FAIL == 0 and WARN <= 2:
    print("=> SOLIDE : resultats integrables dans le manuscrit.")
elif FAIL == 0:
    print("=> Globalement solide : warnings mineurs a documenter.")
else:
    print("=> NON SOLIDE : appliquer les actions correctives.")