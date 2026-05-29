"""
calibrate_bruchovsky.py
========================

Per-patient profile-likelihood calibration of the §7 fractional integral
equation on the Bruchovsky OFF-phase cohort.

This is the baseline Trial C-damped calibration of the paper: one
free fractional order alpha per patient, with the remaining model
parameters fixed at clinically-calibrated values, on the contraction-safe
truncated mesh T_max = 320 d. Picard iteration is damped with relaxation
factor w = 0.5 to ensure convergence at alpha-extremes (without damping,
the operator modulus is 0.97 with a negative spectrum on parts of the
admissible range, leading to oscillating / slow / unstable iterates and
artifactual clamping of alpha_hat).

The companion script `calibrate_bruchovsky_Tmax_invariance.py` re-runs the
same calibration with the extended mesh T_max = 2000 d to verify
empirically that the calibration outcome is invariant to the mesh
truncation, as predicted by the amplitude bound of Theorem 9.1
(which depends on the model parameters only, not on T_max).

Usage:
    python -m src.calibration.calibrate_bruchovsky

Estimated wall time: 3-5 minutes (T_max = 320 is a small mesh).

Outputs (written to ./outputs/):
    bruchovsky_calibration_offphase.csv
    bruchovsky_calibration_stats_offphase.txt
    fig_alpha_recovery_offphase.pdf
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure the numerical_schemes/ sibling package is importable
# whether this script is invoked directly or with `python -m`.
_NUMERICAL_SCHEMES_DIR = Path(__file__).resolve().parents[1] / "numerical_schemes"
if str(_NUMERICAL_SCHEMES_DIR) not in sys.path:
    sys.path.insert(0, str(_NUMERICAL_SCHEMES_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.special import gamma as gamma_fn

from classical_scheme import I_minus_quadrature, distributed_delay


SUFFIX_IN  = "_offphase"
SUFFIX_OUT = "_offphase_v3_damped"

# === KEY CHANGES vs v3 ORIGINAL ===
H = 2.0                # mesh resolution unchanged
T_MAX = 320.0          # SAME as v3 original
N_ITER_MAX = 500       # SAME as v3 original
TOL_PIC = 1e-9         # SAME as v3 original
PICARD_DAMPING = 0.5   # NEW: same as v4
Y_OVERFLOW_THRESHOLD = 1e3

# Standard kernel (same as v3 original)
NUISANCE = {
    "p":         3.0,
    "mu":        0.1,
    "beta_0":    0.0,
    "gamma_sat": 0.6,
    "xi":        0.4,
    "tau":       0.0,
    "T_w":       20.0,
}

K_ALPHA_REF = 0.4431131
L_F = 0.6 + 0.4
A_0_MAX = 2.0

ALPHA_LO, ALPHA_HI = 1.05, 1.95
N_COARSE = 41
BRENT_TOL = 1e-5
H_ALPHA_CURV = 1e-3

REGIME_COLORS = {
    "eradication": "#2E7D32",
    "dormancy":    "#EF6C00",
    "escape":      "#C62828",
}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def output_dir() -> Path:
    p = project_root() / "outputs_clinical"
    p.mkdir(parents=True, exist_ok=True)
    return p


def figuras_dir() -> Path:
    p = project_root() / "figuras"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cohort_csv() -> Path:
    return output_dir() / f"bruchovsky_cohort_filtered{SUFFIX_IN}.csv"


def per_patient_csv() -> Path:
    return output_dir() / f"bruchovsky_per_patient{SUFFIX_IN}.csv"


def corrected_A0(y_first: float, c_0: float) -> float:
    if c_0 <= 1e-6:
        return A_0_MAX
    ratio = max(0.0, min(1.0, y_first / c_0))
    required = (1.0 - ratio) / (K_ALPHA_REF * L_F)
    return float(min(A_0_MAX, required))


def make_forward_solver(A_0: float, c_0: float):
    p = NUISANCE["p"]; mu = NUISANCE["mu"]
    beta_0 = NUISANCE["beta_0"]; gamma_sat = NUISANCE["gamma_sat"]
    xi = NUISANCE["xi"]; tau = NUISANCE["tau"]; T_w = NUISANCE["T_w"]

    def a_func(s): return A_0 * (1.0 + s) ** (-p)
    def f_func(u, v): return beta_0 - gamma_sat * u - xi * v
    def K_func(s, sigma): return mu * np.exp(-mu * (s - sigma))
    def g_func(y_arr):
        if tau == 0.0:
            return c_0
        n_T = max(2, int(round(T_w / H)) + 1)
        n_T = min(n_T, len(y_arr))
        y_w = y_arr[:n_T]
        avg = (H / T_w) * (0.5 * y_w[0] + y_w[1:-1].sum() + 0.5 * y_w[-1])
        return c_0 + tau * avg

    return a_func, f_func, K_func, g_func


def picard_truncated_damped(a, f, K, g, alpha, h, T_max,
                              n_iter=500, tol=1e-9, w=PICARD_DAMPING):
    """Damped Picard iteration: y_{k+1} = (1-w)*y_k + w*T(y_k).

    Same fixed point as undamped Picard, but w=0.5 stabilises the iteration
    against the negative-modulus oscillation when |L| ~ 1 at alpha-extremes.
    """
    N = int(round(T_max / h))
    t = h * np.arange(N + 1)
    a_vals = np.asarray([a(ti) for ti in t], dtype=float)

    if callable(g):
        try:
            g0 = float(g(np.zeros(N + 1)))
        except Exception:
            g0 = 0.0
    else:
        g0 = float(g)
    y = g0 * np.ones(N + 1, dtype=float)

    res_history = []
    iters_done = 0
    for k in range(n_iter):
        iters_done = k + 1
        Ky = distributed_delay(K, y, h) if K is not None else np.zeros_like(y)
        F = np.array([f(y[i], Ky[i]) for i in range(N + 1)], dtype=float)
        aF = a_vals * F
        Iv = I_minus_quadrature(aF, h, alpha)
        g_val = float(g(y)) if callable(g) else float(g)

        y_new = (1.0 - w) * y + w * (g_val + Iv)

        if not np.all(np.isfinite(y_new)) or \
           np.max(np.abs(y_new)) > Y_OVERFLOW_THRESHOLD:
            return t, y, dict(iters=iters_done,
                              res_history=np.array(res_history),
                              status="diverged")

        res = float(np.max(np.abs(y_new - y)))
        res_history.append(res)
        y = y_new
        if res < tol:
            break

    return t, y, dict(iters=iters_done,
                      res_history=np.array(res_history),
                      status="converged" if res_history and res_history[-1] < tol
                             else "max_iter")


def solve_forward(alpha, A_0, c_0):
    a_func, f_func, K_func, g_func = make_forward_solver(A_0, c_0)
    return picard_truncated_damped(
        a_func, f_func, K_func, g_func,
        alpha=alpha, h=H, T_max=T_MAX,
        n_iter=N_ITER_MAX, tol=TOL_PIC, w=PICARD_DAMPING,
    )


def evaluate_at_obs(alpha, A_0, c_0, t_obs):
    t_mesh, y_mesh, info = solve_forward(alpha, A_0, c_0)
    if not np.all(np.isfinite(y_mesh)) or \
       np.max(np.abs(y_mesh)) > Y_OVERFLOW_THRESHOLD:
        return np.full_like(t_obs, np.nan, dtype=float), info["iters"]
    interp = interp1d(t_mesh, y_mesh, kind="linear",
                      bounds_error=False,
                      fill_value=(y_mesh[0], y_mesh[-1]))
    return interp(t_obs), info["iters"]


def loss(alpha, A_0, c_0, t_train, y_train):
    y_pred, _ = evaluate_at_obs(alpha, A_0, c_0, t_train)
    if not np.all(np.isfinite(y_pred)):
        return float("inf")
    res = y_pred - y_train
    return float(np.mean(res ** 2))


def fit_alpha(t_train, y_train, A_0, c_0):
    alpha_grid = np.linspace(ALPHA_LO, ALPHA_HI, N_COARSE)
    L_grid = np.array([loss(a, A_0, c_0, t_train, y_train) for a in alpha_grid])
    j_star = int(np.argmin(L_grid))
    lo = alpha_grid[max(0, j_star - 1)]
    hi = alpha_grid[min(N_COARSE - 1, j_star + 1)]
    if hi - lo < 1e-3:
        return float(alpha_grid[j_star]), float(L_grid[j_star]), alpha_grid, L_grid

    def f(a): return loss(a, A_0, c_0, t_train, y_train)
    res = minimize_scalar(f, bounds=(lo, hi), method="bounded",
                          options={"xatol": BRENT_TOL})
    alpha_hat = float(np.clip(res.x, ALPHA_LO, ALPHA_HI))
    return alpha_hat, float(res.fun), alpha_grid, L_grid


def curvature_L_pp(alpha_hat, A_0, c_0, t_train, y_train):
    h = H_ALPHA_CURV
    if alpha_hat - 2 * h <= 1.0 or alpha_hat + 2 * h >= 2.0:
        L_minus = loss(alpha_hat - h, A_0, c_0, t_train, y_train)
        L_zero  = loss(alpha_hat,     A_0, c_0, t_train, y_train)
        L_plus  = loss(alpha_hat + h, A_0, c_0, t_train, y_train)
        return (L_plus - 2 * L_zero + L_minus) / (h * h)
    L_m2 = loss(alpha_hat - 2*h, A_0, c_0, t_train, y_train)
    L_m1 = loss(alpha_hat -   h, A_0, c_0, t_train, y_train)
    L_z  = loss(alpha_hat,       A_0, c_0, t_train, y_train)
    L_p1 = loss(alpha_hat +   h, A_0, c_0, t_train, y_train)
    L_p2 = loss(alpha_hat + 2*h, A_0, c_0, t_train, y_train)
    return (-L_m2 + 16*L_m1 - 30*L_z + 16*L_p1 - L_p2) / (12*h*h)


def calibrate_one_patient(pid, sub_long, A_0, c_0, regime):
    sub_train = sub_long[sub_long["split"] == "train"].sort_values("t")
    sub_test  = sub_long[sub_long["split"] == "test"].sort_values("t")
    t_train = sub_train["t"].values.astype(float)
    y_train = sub_train["y_norm"].values.astype(float)
    t_test  = sub_test["t"].values.astype(float)
    y_test  = sub_test["y_norm"].values.astype(float)

    if len(t_train) < 4:
        return dict(patient_id=pid, regime=regime, status="skipped_few_train")

    t0 = time.time()
    alpha_hat, L_min, _, _ = fit_alpha(t_train, y_train, A_0, c_0)

    sigma_hat = float(np.sqrt(max(L_min, 1e-16)))
    L_pp = curvature_L_pp(alpha_hat, A_0, c_0, t_train, y_train)
    if L_pp <= 0 or not np.isfinite(L_pp):
        SE_CR = float("nan"); ci_lo = ci_hi = float("nan")
    else:
        SE_CR = sigma_hat * np.sqrt(2.0 / (len(t_train) * L_pp))
        ci_lo = max(ALPHA_LO, alpha_hat - 1.96 * SE_CR)
        ci_hi = min(ALPHA_HI, alpha_hat + 1.96 * SE_CR)

    y_pred_train, n_iter_train = evaluate_at_obs(alpha_hat, A_0, c_0, t_train)
    if np.all(np.isfinite(y_pred_train)):
        train_RMSE = float(np.sqrt(np.mean((y_pred_train - y_train) ** 2)))
    else:
        train_RMSE = float("nan")

    if len(t_test) > 0:
        y_pred_test, _ = evaluate_at_obs(alpha_hat, A_0, c_0, t_test)
        if np.all(np.isfinite(y_pred_test)):
            test_RMSE = float(np.sqrt(np.mean((y_pred_test - y_test) ** 2)))
        else:
            test_RMSE = float("nan")
    else:
        test_RMSE = float("nan")

    return dict(
        patient_id=pid, regime=regime,
        alpha_hat=alpha_hat, L_min=L_min,
        sigma_hat=sigma_hat, L_pp_curvature=L_pp, SE_CR=SE_CR,
        CI95_lo=ci_lo, CI95_hi=ci_hi,
        train_RMSE=train_RMSE, test_RMSE=test_RMSE,
        n_train=len(t_train), n_test=len(t_test),
        n_iter_picard=n_iter_train,
        wall_seconds=time.time() - t0, status="ok",
        A_0_used=A_0, c_0_used=c_0,
    )


def write_stats(df_res, A_0_per_pat):
    p = output_dir() / f"bruchovsky_calibration_stats{SUFFIX_OUT}.txt"
    with open(p, "w") as f:
        f.write("Trial C-damped calibration calibration summary "
                "(OFF-phase, T_max=320 + damped Picard)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Re-calibration of v3 with damped Picard iteration:\n")
        f.write(f"  T_max = {T_MAX} d, n_iter_max = {N_ITER_MAX}, "
                f"tol = {TOL_PIC}, w_damping = {PICARD_DAMPING}\n\n")
        f.write("Theoretical justification:\n")
        f.write(f"  A_0_i = min({A_0_MAX}, (1 - y_first/c_0) / "
                f"({K_ALPHA_REF:.6f} * {L_F}))\n")
        f.write(f"  K_alpha(1.5) = sqrt(pi)/4 = {K_ALPHA_REF:.6f}, L_F = {L_F}\n\n")
        f.write("Global nuisance:\n")
        for k, v in NUISANCE.items():
            f.write(f"  {k:10s} = {v}\n")
        f.write("\n")
        df_ok = df_res[df_res["status"] == "ok"].copy()
        f.write(f"N patients (calibrated): {len(df_ok)}\n\n")

        a0s = list(A_0_per_pat.values())
        f.write(f"A_0 corrected distribution (per-patient):\n")
        f.write(f"  range = [{min(a0s):.3f}, {max(a0s):.3f}]\n")
        f.write(f"  mean = {np.mean(a0s):.3f}, median = {np.median(a0s):.3f}\n")
        n_at_cap = sum(1 for a in a0s if a >= A_0_MAX - 1e-6)
        f.write(f"  Patients at A_0_max cap = {n_at_cap}/{len(a0s)} "
                f"({100*n_at_cap/len(a0s):.1f}%)\n\n")

        f.write("alpha_hat distribution (overall):\n")
        f.write(df_ok["alpha_hat"].describe().to_string()); f.write("\n\n")
        f.write("alpha_hat by regime:\n")
        f.write(df_ok.groupby("regime")["alpha_hat"].describe().to_string())
        f.write("\n\n")
        f.write("train_RMSE by regime:\n")
        f.write(df_ok.groupby("regime")["train_RMSE"].describe().to_string())
        f.write("\n\n")
        f.write("test_RMSE (forecast) by regime:\n")
        f.write(df_ok.groupby("regime")["test_RMSE"].describe().to_string())
        f.write("\n\n")
        f.write("Picard iterations distribution:\n")
        f.write(df_ok["n_iter_picard"].describe().to_string()); f.write("\n\n")
        f.write("Wall-time per patient (s):\n")
        f.write(df_ok["wall_seconds"].describe().to_string()); f.write("\n\n")
        f.write(f"Patients with valid CR SE (L_pp > 0): "
                f"{(df_ok['L_pp_curvature']>0).sum()}/{len(df_ok)}\n\n")
        n_lo = (df_ok["alpha_hat"] <= ALPHA_LO + 0.01).sum()
        n_hi = (df_ok["alpha_hat"] >= ALPHA_HI - 0.01).sum()
        f.write(f"Patients with alpha_hat at clamp:\n"
                f"  lower (<= {ALPHA_LO + 0.01}):  {n_lo}\n"
                f"  upper (>= {ALPHA_HI - 0.01}):  {n_hi}\n")
    print(f"  wrote {p}")


def make_figure(df_res, df_long, per_pat):
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.30)
    df_ok = df_res[df_res["status"] == "ok"].copy()

    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(ALPHA_LO, ALPHA_HI, 19)
    for regime in ["eradication", "dormancy", "escape"]:
        sub = df_ok[df_ok["regime"] == regime]
        ax.hist(sub["alpha_hat"], bins=bins, alpha=0.55,
                color=REGIME_COLORS[regime],
                label=f"{regime} (n={len(sub)})")
    ax.set_xlabel(r"$\hat\alpha$"); ax.set_ylabel("count")
    ax.set_title(r"(a) $\hat\alpha$ by regime (Trial C-damped)", fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    data = [df_ok.loc[df_ok["regime"] == r, "alpha_hat"].values
            for r in ["eradication", "dormancy", "escape"]]
    bplot = ax.boxplot(data, tick_labels=["erad.", "dorm.", "esc."],
                       patch_artist=True, widths=0.55)
    for patch, regime in zip(bplot["boxes"],
                             ["eradication", "dormancy", "escape"]):
        patch.set_facecolor(REGIME_COLORS[regime]); patch.set_alpha(0.55)
    ax.set_ylabel(r"$\hat\alpha$")
    ax.set_title(r"(b) $\hat\alpha$ boxplot (Trial C-damped)", fontsize=10)

    ax = fig.add_subplot(gs[0, 2])
    for regime in ["eradication", "dormancy", "escape"]:
        sub = df_ok[df_ok["regime"] == regime]
        ax.scatter(sub["alpha_hat"], sub["test_RMSE"],
                   color=REGIME_COLORS[regime], alpha=0.65, s=22,
                   label=regime)
    ax.set_xlabel(r"$\hat\alpha$"); ax.set_ylabel("test RMSE")
    ax.set_title(r"(c) Forecast vs $\hat\alpha$ (Trial C-damped)", fontsize=10)
    ax.legend(fontsize=8)

    paradigm_pids = []
    for regime in ["eradication", "dormancy", "escape"]:
        sub = df_ok[df_ok["regime"] == regime]
        if len(sub) > 0:
            sub_sorted = sub.sort_values("test_RMSE")
            paradigm_pids.append(int(sub_sorted.iloc[len(sub_sorted)//2]["patient_id"]))
        else:
            paradigm_pids.append(None)

    for k, pid in enumerate(paradigm_pids):
        ax = fig.add_subplot(gs[1, k])
        if pid is None:
            ax.text(0.5, 0.5, "(n/a)", ha="center", va="center",
                    transform=ax.transAxes); continue
        sub = df_long[df_long["patient_id"] == pid].sort_values("t")
        sub_train = sub[sub["split"] == "train"]
        sub_test  = sub[sub["split"] == "test"]
        regime = per_pat.loc[per_pat["patient_id"] == pid, "regime"].iloc[0]
        c_0 = float(per_pat.loc[per_pat["patient_id"] == pid, "c0_calib"].iloc[0])
        A_0 = float(df_ok.loc[df_ok["patient_id"] == pid, "A_0_used"].iloc[0])
        alpha_hat = float(df_ok.loc[df_ok["patient_id"] == pid, "alpha_hat"].iloc[0])
        t_mesh, y_mesh, _ = solve_forward(alpha_hat, A_0, c_0)
        color = REGIME_COLORS[regime]
        ax.plot(t_mesh, y_mesh, "-", color=color, lw=1.5,
                label=fr"§7 ($\hat\alpha={alpha_hat:.3f}$, $A_0={A_0:.2f}$)")
        ax.plot(sub_train["t"], sub_train["y_norm"], "o",
                color="black", ms=4, label="train")
        ax.plot(sub_test["t"], sub_test["y_norm"], "s",
                color="red", ms=4, label="test")
        ax.set_xlim(0, max(sub["t"].max() * 1.05, 50))
        ax.set_ylim(-0.05, 1.10)
        ax.set_xlabel("t (days from OFF start)")
        ax.set_ylabel(r"$y_{\rm norm}$")
        ax.set_title(f"({chr(100+k)}) Pt {pid} ({regime})", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.25, lw=0.4)

    for k, regime in enumerate(["eradication", "dormancy", "escape"]):
        ax = fig.add_subplot(gs[2, k])
        sub = df_ok[df_ok["regime"] == regime].sort_values("alpha_hat")
        if len(sub) == 0:
            ax.text(0.5, 0.5, "(n/a)", ha="center", va="center",
                    transform=ax.transAxes); continue
        x = np.arange(len(sub))
        color = REGIME_COLORS[regime]
        ax.errorbar(x, sub["alpha_hat"],
                    yerr=[sub["alpha_hat"] - sub["CI95_lo"].fillna(sub["alpha_hat"]),
                          sub["CI95_hi"].fillna(sub["alpha_hat"]) - sub["alpha_hat"]],
                    fmt="o", color=color, ms=3, elinewidth=0.8,
                    capsize=2, alpha=0.85)
        ax.set_xlabel("patient (sorted by $\\hat\\alpha$)")
        ax.set_ylabel(r"$\hat\alpha \pm 1.96\,\mathrm{SE}_\mathrm{CR}$")
        ax.set_title(f"({chr(103+k)}) {regime} (n={len(sub)})", fontsize=10)
        ax.set_ylim(ALPHA_LO - 0.05, ALPHA_HI + 0.05)
        ax.grid(True, alpha=0.25, lw=0.4)

    fig.suptitle(rf"Trial C-damped calibration (OFF-phase, $T_{{\max}}=320$ d, "
                 r"damped Picard $w=0.5$): $\hat\alpha$ recovery",
                 fontsize=11)
    p = figuras_dir() / f"fig_S9_2_alpha_recovery{SUFFIX_OUT}.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Trial C-damped (T_max=320 + damped Picard) — usage:\n"
                    "  smoke (3 patients):  python3 ... --max-patients 3\n"
                    "  full  (all 55):      python3 ...")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--start-patient", type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print("Trial C-damped calibration (OFF-phase, T_max=320 + damped Picard)")
    print("=" * 60)
    if args.max_patients is not None:
        print(f"  >>> SUBSET MODE: first {args.max_patients} patients <<<")

    print(f"\nGlobal nuisance: {NUISANCE}")
    print(f"T_max = {T_MAX} d, h = {H} d, n_iter_max = {N_ITER_MAX}, "
          f"tol = {TOL_PIC}, w_damping = {PICARD_DAMPING}")
    print(f"A_0 corrected per patient: A_0_i = min({A_0_MAX}, "
          f"(1 - y_first/c_0)/({K_ALPHA_REF:.4f}*{L_F}))")

    print("\n[load] reading OFF-phase cohort...")
    if not cohort_csv().exists():
        sys.exit(f"OFF-phase cohort not found at {cohort_csv()}")
    df = pd.read_csv(cohort_csv())
    per_pat = pd.read_csv(per_patient_csv())
    n_pat = per_pat.shape[0]
    print(f"  {n_pat} patients, {len(df)} total OFF-phase measurements")

    print("\n[A_0 correction] computing per-patient A_0...")
    A_0_per_pat = {}
    for _, row in per_pat.iterrows():
        pid = int(row["patient_id"])
        c_0 = float(row["c0_calib"])
        sub = df[df["patient_id"] == pid].sort_values("t")
        y_first = float(sub["y_norm"].iloc[0]) if len(sub) > 0 else 0.0
        A_0_per_pat[pid] = corrected_A0(y_first, c_0)
    a0s = list(A_0_per_pat.values())
    print(f"  A_0 distribution: min={min(a0s):.3f}, max={max(a0s):.3f}, "
          f"mean={np.mean(a0s):.3f}")
    n_at_cap = sum(1 for a in a0s if a >= A_0_MAX - 1e-6)
    print(f"  Patients at A_0_max ({A_0_MAX}) cap: {n_at_cap}/{n_pat} "
          f"({100*n_at_cap/n_pat:.1f}%)")

    per_pat_sel = per_pat.iloc[args.start_patient:]
    if args.max_patients is not None:
        per_pat_sel = per_pat_sel.head(args.max_patients)
    n_to_run = len(per_pat_sel)
    print(f"\n  Running on {n_to_run} of {n_pat} patients.")

    results = []
    print(f"\n[calibrate] running per-patient profile-likelihood ...")
    print(f"  Mesh size: {int(T_MAX/H) + 1} nodes (small).")
    print(f"  Expected wall time: ~5-15 sec per patient, "
          f"~{n_to_run * 10 // 60} min total.\n")
    t0_all = time.time()
    for i, row in per_pat_sel.iterrows():
        pid = int(row["patient_id"])
        A_0 = A_0_per_pat[pid]
        c_0 = float(row["c0_calib"])
        regime = row["regime"]
        sub = df[df["patient_id"] == pid]
        out = calibrate_one_patient(pid, sub, A_0, c_0, regime)
        results.append(out)
        elapsed = time.time() - t0_all
        n_done = i + 1
        ETA = elapsed * (n_to_run - n_done) / n_done if n_done > 0 else 0
        print(f"  [{n_done:2d}/{n_pat}] pid={pid:3d} regime={regime:11s} "
              f"A_0={A_0:.3f} alpha_hat={out.get('alpha_hat', float('nan')):.3f} "
              f"train_RMSE={out.get('train_RMSE', float('nan')):.4f} "
              f"test_RMSE={out.get('test_RMSE', float('nan')):.4f} "
              f"n_iter={out.get('n_iter_picard', '?')} "
              f"({elapsed:.0f}s, ETA {ETA:.0f}s)")

    df_res = pd.DataFrame(results)
    csv_path = output_dir() / f"bruchovsky_calibration{SUFFIX_OUT}.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n[output] wrote {csv_path}")

    print("\n[stats] computing aggregate metrics...")
    write_stats(df_res, A_0_per_pat)

    print("\n[figure] rendering S9.2 Trial C-damped multi-panel...")
    make_figure(df_res, df, per_pat)

    print("\n" + "=" * 60)
    elapsed = time.time() - t0_all
    print(f"Done. {n_to_run} patients calibrated in {elapsed:.1f} s "
          f"({elapsed/60:.1f} min).")
    print("\nNEXT: run the T_max-invariance control (T_max = 2000 d) to verify")
    print("      that the calibration outcome is independent of mesh truncation:")
    print("  python -m src.calibration.calibrate_bruchovsky_Tmax_invariance")
    print("=" * 60)


if __name__ == "__main__":
    main()
