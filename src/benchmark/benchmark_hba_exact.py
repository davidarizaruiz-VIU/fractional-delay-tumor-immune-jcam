"""
benchmark_hba_exact.py
=======================

Head-to-head numerical benchmark on the Bruchovsky cohort (cycle 1 OFF-phase):

  §7 framework (Trial C-damped, 1 calibrated alpha + global nuisance)
       VS
  Hirata-Bruchovsky-Aihara (2010) EXACT 3-compartment model, OFF-phase

The HBA-2010 model (Hirata, Bruchovsky, Aihara, J Theor Biol 264:517-527, 2010)
is a piecewise-linear three-compartment ODE with separate equations for the
treatment (ON) and no-treatment (OFF) periods. For our cycle 1 OFF-phase
analysis, only the OFF-period equation (eq. 2 of the paper) is needed:

  d/dt [x_1; x_2; x_3] = [w_11  w_12  0  ;
                           0    w_22  0  ;
                           0    0     w_33] [x_1; x_2; x_3]

  PSA(t) = x_1(t) + x_2(t) + x_3(t)        (eq. 24 of the paper)

where:
  x_1 = androgen-dependent (AD) cells
  x_2 = androgen-independent reversible (AI rev) cells
  x_3 = androgen-independent irreversible (AI irrev) cells

  w_11 > 0:  AD growth rate (AD cells re-grow under no-treatment)
  w_12 > 0:  reflux rate from AI rev to AD (reversible adaptation)
  w_22:      AI rev evolution rate (sign-free)
  w_33:      AI irrev evolution rate (sign-free, often > 0 → relapse)

Free parameters per patient (OFF cycle 1 only): 4 rates + 3 initial conditions
= 7 parameters. (The full HBA paper has 9 parameters per patient including
ON-phase rates, but the OFF cycle only excites these 7.)

Per the paper's Section 4.1, the ODE is solved by Euler discretisation
with Δt=1 day. We follow this exact convention here for fidelity to the
original paper.

Calibration: Levenberg-Marquardt (scipy.optimize.least_squares with trust-region
reflective method, ftol = xtol = 1e-10), on the same 70/30 chronological
train/test split as the §7 calibration.

Outputs:
    outputs_clinical/benchmark_hba_exact.csv
    outputs_clinical/benchmark_hba_exact_stats.txt
    figuras/fig_S9_benchmark_hba_exact.pdf
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


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


REGIME_COLORS = {
    "eradication": "#2E7D32",
    "dormancy":    "#EF6C00",
    "escape":      "#C62828",
}


# ---------------------------------------------------------------------------
# HBA-2010 exact OFF-phase forward solver (Euler discretisation, Δt=1 day)
# ---------------------------------------------------------------------------


def hba_off_psa(t_obs, w11, w12, w22, w33, x1_0, x2_0, x3_0):
    """Compute PSA(t) = x_1(t) + x_2(t) + x_3(t) at observation times t_obs.

    Solver: Euler discretisation with Δt = 1 day, exactly as Section 4.1 of
    Hirata-Bruchovsky-Aihara 2010 (eq. 11):

      x_{k+1} = (I + Δt · W^0) x_k

    where W^0 is the OFF-phase rate matrix (eq. 2 of the paper).
    """
    t_obs = np.asarray(t_obs, dtype=float)
    if len(t_obs) == 0:
        return np.array([])
    # Round t_obs to nearest integer day, since the HBA paper uses Δt = 1 day
    t_int = np.round(t_obs).astype(int)
    t_max = int(max(t_int.max(), 1))

    # Closed-form Euler propagation: M is upper-triangular, so x(t) has
    # closed form via decoupling. Specifically:
    #   x_3(k) = (1+w33)^k * x3_0
    #   x_2(k) = (1+w22)^k * x2_0
    #   x_1(k) = (1+w11)^k * x1_0 + w12 * sum_{j=0}^{k-1} (1+w11)^{k-1-j} (1+w22)^j * x2_0
    # The sum is a geometric series; closed form:
    #   sum = ((1+w22)^k - (1+w11)^k) / (w22 - w11)   if w11 != w22
    #       = k * (1+w11)^{k-1}                       if w11 = w22
    # PSA(k) = x_1(k) + x_2(k) + x_3(k)
    a, b, c = 1.0 + w11, 1.0 + w22, 1.0 + w33
    # Quick sanity: if any base is non-positive (Euler step blew up), return nan
    if a <= 0 or b <= 0 or c <= 0:
        return np.full_like(t_obs, np.nan)
    # Vectorised over t_int, using log-space to prevent overflow
    # x_3(k) = c^k * x3_0
    log_a, log_b, log_c = np.log(a), np.log(b), np.log(c)
    # Cap k * log(base) to avoid overflow (np.exp(700) ~ 1e304)
    if max(log_a, log_b, log_c) * t_max > 700:
        return np.full_like(t_obs, np.nan)
    k = t_int.astype(float)
    a_pow_k = np.exp(k * log_a)
    b_pow_k = np.exp(k * log_b)
    c_pow_k = np.exp(k * log_c)
    x1_k = a_pow_k * x1_0
    x2_k = b_pow_k * x2_0
    x3_k = c_pow_k * x3_0
    # Coupling term: w12 * sum_{j=0}^{k-1} a^{k-1-j} b^j * x2_0
    # = w12 * x2_0 * (b^k - a^k) / (b - a)   if a != b (within tolerance)
    if abs(a - b) > 1e-10:
        coupling = w12 * x2_0 * (b_pow_k - a_pow_k) / (b - a)
    else:
        # degenerate case
        coupling = w12 * x2_0 * k * a_pow_k / a
    x1_k = x1_k + coupling
    psa = x1_k + x2_k + x3_k
    if not np.all(np.isfinite(psa)) or np.max(np.abs(psa)) > 1e15:
        return np.full_like(t_obs, np.nan)
    return psa


def calibrate_hba_one_patient(t_train, y_train, t_test, y_test,
                               c_0, y_first):
    """Calibrate (w11, w12, w22, w33, x1_0, x2_0, x3_0) on the training set,
       then evaluate forecast (test RMSE) on the test set.

    Initial guess: simple growth model
      y(t) ≈ y_first + (c_0 - y_first)(1 - exp(-r*t))
    Estimate r from time to half-rebound. Distribute initial PSA as
    x_1(0) ≈ y_first (most AD), x_2(0) ≈ 0, x_3(0) ≈ 0.

    Bounds (from HBA Section 4.2 and our OFF-phase setting):
      growth/transition rates in [-0.005, 0.10] per day
        (allows decline at -0.5%/day up to 10%/day growth)
      initial conditions in [0, 5] (normalised PSA, max ~1 + slack)
    """
    if len(t_train) < 5:
        return dict(status="skipped_few_train")

    # Initial guess for growth rate from data
    target = y_first + 0.5 * (c_0 - y_first)
    above = np.where(y_train >= target)[0]
    if len(above) > 0:
        t_half = max(t_train[above[0]], 1.0)
        r0 = float(np.log(2) / t_half)
    else:
        r0 = 0.005
    r0 = max(min(r0, 0.05), 0.001)  # clamp to reasonable range

    # 7 parameters: (w11, w12, w22, w33, x1_0, x2_0, x3_0)
    # Tighter bounds: small growth rates (≤ 5%/day) to prevent extrapolation explosion;
    # x_i(0) bounded by [0, 1.5] (normalised PSA range with slack)
    bounds = (
        [-0.005, 0.0,  -0.005, -0.005, 0.0,  0.0,  0.0],
        [ 0.05,  0.05,  0.05,   0.05,  1.5,  1.5,  1.5],
    )

    # Multi-start: try several initial conditions to escape local minima
    initial_guesses = [
        # (1) Standard guess: AD-dominant initial
        np.array([r0, 0.001, 0.5*r0, 0.5*r0, max(y_first, 1e-3), 1e-3, 1e-3]),
        # (2) Mixed initial: equal AD/AI populations
        np.array([r0, 0.005, r0, r0, y_first/3, y_first/3, y_first/3]),
        # (3) Slow AD growth, larger irrev
        np.array([r0/2, 0.002, r0/2, r0, y_first/2, 1e-3, y_first/2]),
        # (4) AI-dominant, slow AD
        np.array([r0/4, 0.01, r0, r0/2, 1e-3, y_first, 1e-3]),
    ]

    t0 = time.time()

    def residual(p):
        y_pred = hba_off_psa(t_train, *p)
        if not np.all(np.isfinite(y_pred)):
            return np.full_like(t_train, 1e3)
        return y_pred - y_train

    best_result = None
    best_cost = float("inf")
    for p0 in initial_guesses:
        # Clip p0 to bounds
        p0 = np.clip(p0, bounds[0], bounds[1])
        try:
            result = least_squares(residual, p0, bounds=bounds, method="trf",
                                    ftol=1e-8, xtol=1e-8, max_nfev=500)
            if result.success and result.cost < best_cost:
                best_cost = result.cost
                best_result = result
        except Exception:
            continue

    try:
        if best_result is None:
            return dict(status="optim_failed", message="all multistart failed")
        result = best_result

        params = result.x
        y_pred_train = hba_off_psa(t_train, *params)
        if np.all(np.isfinite(y_pred_train)):
            train_RMSE = float(np.sqrt(np.mean((y_pred_train - y_train) ** 2)))
        else:
            train_RMSE = float("nan")

        if len(t_test) > 0:
            y_pred_test = hba_off_psa(t_test, *params)
            if np.all(np.isfinite(y_pred_test)):
                test_RMSE = float(np.sqrt(np.mean((y_pred_test - y_test) ** 2)))
            else:
                test_RMSE = float("nan")
        else:
            test_RMSE = float("nan")

        return dict(
            status="ok",
            w11=float(params[0]), w12=float(params[1]),
            w22=float(params[2]), w33=float(params[3]),
            x1_0=float(params[4]), x2_0=float(params[5]), x3_0=float(params[6]),
            train_RMSE=train_RMSE, test_RMSE=test_RMSE,
            n_train=len(t_train), n_test=len(t_test),
            wall_seconds=time.time() - t0,
        )
    except Exception as e:
        return dict(status="exception", message=str(e))


def main() -> None:
    print("=" * 70)
    print("HBA-2010 EXACT 3-compartment benchmark on Bruchovsky OFF-phase")
    print("=" * 70)

    print("\n[load] reading OFF-phase cohort...")
    df = pd.read_csv(output_dir() / "bruchovsky_cohort_filtered_offphase.csv")
    per_pat = pd.read_csv(output_dir() / "bruchovsky_per_patient_offphase.csv")
    n_pat = per_pat.shape[0]
    print(f"  {n_pat} patients, {len(df)} total OFF-phase measurements")

    print("\n[calibrate] running HBA-2010 OFF-phase exact calibration ...")
    print(f"  Each patient: 7 free params (w11, w12, w22, w33, x1(0), x2(0), x3(0))")
    print(f"  Solver: Euler Δt=1 day (per HBA paper Section 4.1)")
    print(f"  Calibration: Levenberg-Marquardt (scipy.least_squares trust region)\n")

    results = []
    t0_all = time.time()
    for i, row in per_pat.iterrows():
        pid = int(row["patient_id"])
        regime = row["regime"]
        c_0 = float(row["c0_calib"])
        sub = df[df["patient_id"] == pid].sort_values("t")
        sub_train = sub[sub["split"] == "train"]
        sub_test = sub[sub["split"] == "test"]
        t_train = sub_train["t"].values.astype(float)
        y_train = sub_train["y_norm"].values.astype(float)
        t_test = sub_test["t"].values.astype(float)
        y_test = sub_test["y_norm"].values.astype(float)
        y_first = float(sub["y_norm"].iloc[0])

        out = calibrate_hba_one_patient(t_train, y_train, t_test, y_test,
                                          c_0, y_first)
        out["patient_id"] = pid
        out["regime"] = regime
        results.append(out)
        elapsed = time.time() - t0_all
        if (i + 1) % 5 == 0 or i == n_pat - 1:
            n_done = i + 1
            ETA = elapsed * (n_pat - n_done) / n_done if n_done > 0 else 0
            train_R = out.get("train_RMSE", float("nan"))
            test_R = out.get("test_RMSE", float("nan"))
            print(f"  [{n_done:3d}/{n_pat}] pid={pid:3d} regime={regime:11s} "
                  f"status={out.get('status','?'):10s} "
                  f"train_RMSE={train_R:.4f} test_RMSE={test_R:.4f} "
                  f"({elapsed:.0f}s, ETA {ETA:.0f}s)")

    df_res = pd.DataFrame(results)
    csv_path = output_dir() / "benchmark_hba_exact.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n[output] wrote {csv_path}")

    # Compare with Trial C-damped (paper nomenclature; CSV is v3_damped)
    df_ok = df_res[df_res["status"] == "ok"].copy()
    print(f"\n[stats] {len(df_ok)} of {n_pat} patients calibrated successfully")

    v3d = pd.read_csv(output_dir() / "bruchovsky_calibration_offphase_v3_damped.csv")
    v3d_ok = v3d[v3d["status"] == "ok"][["patient_id", "regime",
                                          "train_RMSE", "test_RMSE"]]
    v3d_ok = v3d_ok.rename(columns={"train_RMSE": "train_RMSE_S7",
                                    "test_RMSE": "test_RMSE_S7"})

    df_compare = df_ok[["patient_id", "regime", "train_RMSE", "test_RMSE",
                        "w11", "w12", "w22", "w33",
                        "x1_0", "x2_0", "x3_0"]].rename(
        columns={"train_RMSE": "train_RMSE_HBA",
                 "test_RMSE": "test_RMSE_HBA"})
    df_compare = df_compare.merge(
        v3d_ok[["patient_id", "train_RMSE_S7", "test_RMSE_S7"]],
        on="patient_id"
    )

    stats_path = output_dir() / "benchmark_hba_exact_stats.txt"
    with open(stats_path, "w") as f:
        f.write("HBA-2010 EXACT 3-compartment benchmark vs §7 Trial C-damped\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"N patients (both methods converged): {len(df_compare)}\n\n")

        f.write("HBA-2010 OFF-phase model (eq. 2 of Hirata et al., J Theor Biol 264:517-527):\n")
        f.write("  d/dt [x1; x2; x3] = [[w11, w12, 0], [0, w22, 0], [0, 0, w33]] [x1; x2; x3]\n")
        f.write("  PSA(t) = x1(t) + x2(t) + x3(t)\n")
        f.write("  Solver: Euler Δt=1 day (per HBA Section 4.1)\n")
        f.write("  7 parameters per patient (w11, w12, w22, w33, x1(0), x2(0), x3(0))\n")
        f.write("  Calibration: Levenberg-Marquardt on 70% chronological train\n\n")
        f.write("§7 Trial C-damped: 1 alpha per patient + global nuisance\n")
        f.write("  Calibration: Picard profile-likelihood on same train\n\n")

        for col_HBA, col_S7, label in [("train_RMSE_HBA", "train_RMSE_S7", "Train RMSE"),
                                         ("test_RMSE_HBA", "test_RMSE_S7", "Test RMSE")]:
            f.write(f"--- {label} ---\n")
            f.write(f"  HBA exact:    median = {df_compare[col_HBA].median():.4f}, "
                    f"mean = {df_compare[col_HBA].mean():.4f}, "
                    f"std = {df_compare[col_HBA].std():.4f}\n")
            f.write(f"  §7 Trial C-damped: median = {df_compare[col_S7].median():.4f}, "
                    f"mean = {df_compare[col_S7].mean():.4f}, "
                    f"std = {df_compare[col_S7].std():.4f}\n")
            ratio = (df_compare[col_HBA] / df_compare[col_S7]).replace([np.inf, -np.inf], np.nan)
            f.write(f"  Ratio HBA/S7: median = {ratio.median():.4f}\n")
            wins = (df_compare[col_HBA] < df_compare[col_S7]).sum()
            f.write(f"  Patients where HBA < §7: {wins}/{len(df_compare)} "
                    f"({100*wins/len(df_compare):.1f}%)\n\n")

        f.write("By regime (test RMSE):\n")
        for regime in ["eradication", "dormancy", "escape"]:
            sub = df_compare[df_compare["regime"] == regime]
            if len(sub) == 0: continue
            f.write(f"  {regime} (n={len(sub)}):\n")
            f.write(f"    test_RMSE_HBA median: {sub['test_RMSE_HBA'].median():.4f}\n")
            f.write(f"    test_RMSE_S7  median: {sub['test_RMSE_S7'].median():.4f}\n")
        f.write("\n")

        # Outliers — RIGOROUS reporting:
        # (a) within converged sub-cohort df_compare (denom = len(df_compare))
        # (b) cohort-wide failure rate (HBA non-conv counted as forecast failure)
        n_total = n_pat  # full cohort size (=55)
        n_HBA_failed = int((df_res["status"] != "ok").sum())
        n_S7_failed = int((v3d["status"] != "ok").sum())
        n_HBA_above_05 = int((df_compare["test_RMSE_HBA"] > 0.5).sum())
        n_S7_above_05 = int((df_compare["test_RMSE_S7"] > 0.5).sum())
        n_compare = len(df_compare)
        f.write(f"Outliers within converged sub-cohort (test RMSE > 0.5):\n")
        f.write(f"  HBA exact: {n_HBA_above_05}/{n_compare} "
                f"({100*n_HBA_above_05/n_compare:.1f}%)\n")
        f.write(f"  §7:        {n_S7_above_05}/{n_compare} "
                f"({100*n_S7_above_05/n_compare:.1f}%)\n")
        f.write(f"  HBA test_RMSE max: {df_compare['test_RMSE_HBA'].max():.4f}\n")
        f.write(f"  §7  test_RMSE max: {df_compare['test_RMSE_S7'].max():.4f}\n\n")
        f.write(f"Cohort-wide failure rate (non-convergence + RMSE > 0.5), N = {n_total}:\n")
        f.write(f"  HBA exact: {n_HBA_failed} non-conv + {n_HBA_above_05} RMSE>0.5 "
                f"= {n_HBA_failed + n_HBA_above_05}/{n_total} "
                f"({100*(n_HBA_failed + n_HBA_above_05)/n_total:.1f}%)\n")
        f.write(f"  §7:        {n_S7_failed} non-conv + {n_S7_above_05} RMSE>0.5 "
                f"= {n_S7_failed + n_S7_above_05}/{n_total} "
                f"({100*(n_S7_failed + n_S7_above_05)/n_total:.1f}%)\n")
    print(f"[stats] wrote {stats_path}")

    # Figure: side-by-side comparison with log-y on first panel to reveal outliers
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))

    # Panel (a): scatter test_RMSE on LOG-LOG axes (handles 4 orders of magnitude)
    ax = axes[0]
    for regime in ["eradication", "dormancy", "escape"]:
        sub = df_compare[df_compare["regime"] == regime]
        ax.scatter(sub["test_RMSE_S7"], sub["test_RMSE_HBA"],
                   color=REGIME_COLORS[regime], alpha=0.75, s=36,
                   label=regime, edgecolor="white", linewidth=0.4)
    # Reference y=x line
    rmin = max(min(df_compare["test_RMSE_S7"].min(), df_compare["test_RMSE_HBA"].min()),
                1e-2)
    rmax = max(df_compare["test_RMSE_S7"].max(), df_compare["test_RMSE_HBA"].max())
    ax.plot([rmin, rmax], [rmin, rmax], "--", color="gray", lw=0.8, label="$y=x$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"§7 Trial C-damped test RMSE (log scale)")
    ax.set_ylabel(r"HBA-2010 exact test RMSE (log scale)")
    ax.set_title("(a) per-patient test RMSE comparison", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25, which="both")

    # Panel (b): boxplot test_RMSE — use log scale to show outliers
    ax = axes[1]
    data = [df_compare["test_RMSE_S7"].values, df_compare["test_RMSE_HBA"].values]
    bplot = ax.boxplot(data, tick_labels=["§7\nTrial C-damped\n(1 param)",
                                           "HBA-2010\nexact\n(7 params)"],
                        patch_artist=True, widths=0.55)
    bplot["boxes"][0].set_facecolor("#1976D2"); bplot["boxes"][0].set_alpha(0.7)
    bplot["boxes"][1].set_facecolor("#7B1FA2"); bplot["boxes"][1].set_alpha(0.7)
    ax.set_yscale("log")
    ax.set_ylabel("test RMSE (log scale)")
    ax.set_title("(b) cohort-wide test RMSE distribution", fontsize=10)
    # Annotate outlier counts (within converged sub-cohort) + cohort-wide failure rate
    n_HBA_above_05 = int((df_compare["test_RMSE_HBA"] > 0.5).sum())
    n_S7_above_05 = int((df_compare["test_RMSE_S7"] > 0.5).sum())
    n_HBA_failed = int((df_res["status"] != "ok").sum())
    n_S7_failed = int((v3d["status"] != "ok").sum())
    ax.text(0.02, 0.95,
            f"§7 max: {df_compare['test_RMSE_S7'].max():.2f}\n"
            f"HBA max: {df_compare['test_RMSE_HBA'].max():.0f}\n"
            f"§7 outliers > 0.5: {n_S7_above_05}/{len(df_compare)}\n"
            f"HBA outliers > 0.5: {n_HBA_above_05}/{len(df_compare)}\n"
            f"§7 cohort-wide bad: {n_S7_failed+n_S7_above_05}/{n_pat}\n"
            f"HBA cohort-wide bad: {n_HBA_failed+n_HBA_above_05}/{n_pat}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", lw=0.5))
    ax.grid(True, alpha=0.25, axis="y", which="both")

    # Panel (c): ratio histogram (log scale)
    ax = axes[2]
    ratio = (df_compare["test_RMSE_HBA"] / df_compare["test_RMSE_S7"]).replace(
        [np.inf, -np.inf], np.nan).dropna()
    ax.hist(np.log10(ratio), bins=20, alpha=0.75, color="purple",
            edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="gray", linestyle="--", lw=1.0,
               label="HBA = §7 (ratio = 1)")
    med = float(np.log10(ratio.median()))
    ax.axvline(med, color="darkorange", linestyle="-", lw=1.2,
               label=f"median ratio = {ratio.median():.2f}")
    ax.set_xlabel(r"$\log_{10}$(HBA RMSE / §7 RMSE)")
    ax.set_ylabel("patient count")
    ax.set_title("(c) per-patient RMSE ratio", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle(r"Head-to-head benchmark: §7 fractional framework vs Hirata-Bruchovsky-Aihara (2010) "
                 r"3-compartment ODE (Bruchovsky OFF-phase cycle 1)",
                 fontsize=11)
    fig.tight_layout()

    out_pdf = figuras_dir() / "fig_S9_benchmark_hba_exact.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] wrote {out_pdf}")

    print("\n" + "=" * 70)
    elapsed = time.time() - t0_all
    print(f"Done. {n_pat} patients processed in {elapsed:.1f} s "
          f"({elapsed/60:.1f} min).")
    print("=" * 70)


if __name__ == "__main__":
    main()
