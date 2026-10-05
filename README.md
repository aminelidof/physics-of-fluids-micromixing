# TRIZ-guided magnetic micromixing in lab-on-a-chip devices — PINN assessment

Authors: Mohamed El Amine Fodil1,2,, Merwan Abdelbari3, Meriem Fodil3
1 Department of Hydraulics, Maghnia University Centre, Tlemcen, Algeria
2 Laboratoire Ingénierie et Sciences Appliquées (IScApp), Maghnia, Tlemcen, Algeria
3 Department of Mechanics, Hassiba Ben Bouali University, Chlef, Algeria
Corresponding Author Email: fodilmedam@gmail.com

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PINN DeepXDE](https://img.shields.io/badge/PINN-DeepXDE-orange.svg)](https://github.com/lululxvi/deepxde)

Code and data for the manuscript:

> **"Resolving the productivity–harmful-factors contradiction in lab-on-a-chip micromixing: TRIZ-guided intensification of transverse dispersion assessed by physics-informed neural networks"** *(submitted to Physics of Fluids)*

---

## Summary

A TRIZ technical contradiction — **productivity (parameter 39) vs internally generated harmful factors (parameter 31)** — is resolved by **inventive principle 24 (Intermediary)**: superparamagnetic nanoparticles, agitated by an oscillating external magnetic field, act as a contactless intermediary that intensifies the transverse dispersion by a factor $\kappa$, without any increase in driving pressure[cite: 1].

The steady advection-dispersion model with a Poiseuille profile ($\mathrm{Pe} = 10$) is solved by a **physics-informed neural network (DeepXDE)** and independently by a time-marching finite-difference solver[cite: 1].

---

## Key Results

| Quantity | Value |
| :--- | :--- |
| **Relative $L_2$ error** (PINN vs FDM, 5 runs)[cite: 2] | $1.34\% \pm 0.00$[cite: 2] |
| **RMSE / MAE**[cite: 2] | $6.7\mathrm{e}{-3} / 6.0\mathrm{e}{-3}$[cite: 2] |
| **$R^2$** (full field / mixing region)[cite: 2] | $0.984 / 0.9996$[cite: 2] |
| **Mixing-time reduction**[cite: 2] | $1.29\text{ s} \to 0.26\text{ s}$ (gain $80\%$), pressure unchanged[cite: 2] |
| **Parametric map** ($\mathrm{Pe} \in [2.5, 20] \times \kappa \in [1, 8]$)[cite: 2] | gain $32\text{--}84\%$ (vs bound $1 - 1/\kappa$)[cite: 2] |
| **Robustness** ($N_o = 200, \sigma_n \le 10\%$)[cite: 2] | $L_2 < 1.2\%$; $L_{\text{mix}}$ varies $< 1\%$[cite: 2] |
| **Inverse identification of $\kappa$**[cite: 2] | $\hat{\kappa} = 7.96$ ($0.5\%$ error, noise-free)[cite: 2] |
| **Verification**[cite: 2] | residual $4.2\mathrm{e}{-5}$; conservation $5.9\mathrm{e}{-3}$; grid/collocation-independent[cite: 2] |
| **Quality verdict** (`check_quality.py`)[cite: 2] | 19 OK, 1 WARN, 0 FAIL[cite: 2] |

---

## Repository Structure

```text
├── src/                # Finite-difference solver and evaluation scripts + DeepXDE PINN architecture and training scripts + ├                         Verification and validation quality check script + check_quality.py 
└── README.md
