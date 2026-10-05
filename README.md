TRIZ-guided magnetic micromixing in lab-on-a-chip devices — PINN assessment
License: MITPython 3.10DeepXDE

Code and data for the manuscript:

"Resolving the productivity–harmful-factors contradiction in lab-on-a-chipmicromixing: TRIZ-guided intensification of transverse dispersion assessedby physics-informed neural networks"(submitted to Physics of Fluids)

Summary
A TRIZ technical contradiction — productivity (parameter 39) vsinternally generated harmful factors (parameter 31) — is resolved byinventive principle 24 (Intermediary): superparamagnetic nanoparticles,agitated by an oscillating external magnetic field, act as a contactlessintermediary that intensifies the transverse dispersion by a factor κ,without any increase in driving pressure.

The steady advection–dispersion model with a Poiseuille profile (Pe = 10) issolved by a physics-informed neural network (DeepXDE) and independentlyby a time-marching finite-difference solver.

Key results
Quantity	Value
Relative L₂ error (PINN vs FDM, 5 runs)	1.34% ± 0.00
RMSE / MAE	6.7e-3 / 6.0e-3
R² (full field / mixing region)	0.984 / 0.9996
Mixing-time reduction	1.29 s → 0.26 s (gain 80%), pressure unchanged
Parametric map (Pe ∈ [2.5, 20] × κ ∈ [1, 8])	gain 32–84% (vs bound 1 − 1/κ)
Robustness (N_o = 200, σ_n ≤ 10%)	L₂ < 1.2%; L_mix varies < 1%
Inverse identification of κ	κ̂ = 7.96 (0.5% error, noise-free)
Verification	residual 4.2e-5; conservation 5.9e-3; grid/collocation-independent
Quality verdict (check_quality.py)	19 OK, 1 WARN, 0 FAIL