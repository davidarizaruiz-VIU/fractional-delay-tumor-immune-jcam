"""
make_cohort_amplitude_figure.py
================================

Generate fig_S9_cohort_amplitude_overview.pdf, the central visualisation
of the §9 amplitude bound vs the Bruchovsky cohort.

Three panels:
  (a) Histogram of data amplitude ratio r_i = 1 - y_first/c_0_i across the
      55-patient cohort, colour-coded by clinical regime, with the two
      key thresholds from Theorem 9.1 superimposed:
        - cap-feasibility threshold A_0_cap * K_alpha(1.5) * L_F = 0.886
        - global representable bound K_alpha(1.5)/K_alpha_max = 0.904
  (b) Scatter of A_0_i (per-patient corrected amplitude) vs r_i, with the
      cap line A_0 = 2.0 marking saturation.
  (c) Per-patient ranked diagnostic: r_i sorted descending, classified by
      whether it falls within the framework's representable range or
      exceeds the structural bound.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import beta as Beta, gamma as Gamma


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def output_dir() -> Path:
    return project_root() / "outputs_clinical"


def figuras_dir() -> Path:
    p = project_root() / "figuras"
    p.mkdir(parents=True, exist_ok=True)
    return p


REGIME_COLORS = {
    "eradication": "#2E7D32",
    "dormancy":    "#EF6C00",
    "escape":      "#C62828",
}

# Theoretical thresholds (cf. Theorem 9.1 and §9.2, three-trial demonstration)
ALPHA_NOMINAL = 1.5
P_KERNEL = 3
L_F = 1.0
K_ALPHA_NOMINAL = float(Beta(ALPHA_NOMINAL, P_KERNEL - ALPHA_NOMINAL) /
                         Gamma(ALPHA_NOMINAL))   # = sqrt(pi)/4 = 0.4431
K_ALPHA_MAX = float(Beta(1.05, P_KERNEL - 1.05) /
                     Gamma(1.05))                 # ≈ 0.4899
A_0_CAP = 2.0
THRESHOLD_CAP = A_0_CAP * K_ALPHA_NOMINAL * L_F   # = 0.886
THRESHOLD_GLOBAL = K_ALPHA_NOMINAL * L_F / K_ALPHA_MAX   # ≈ 0.904


def load_cohort_data():
    """Return (r_arr, A0_arr, regime_arr) per-patient."""
    filt = pd.read_csv(output_dir() / "bruchovsky_cohort_filtered_offphase.csv")
    per_pat = pd.read_csv(output_dir() / "bruchovsky_per_patient_offphase.csv")
    v3d = pd.read_csv(output_dir() / "bruchovsky_calibration_offphase_v3_damped.csv")

    # Per-patient first y_norm and c0
    y0 = filt.sort_values(["patient_id", "t"]).groupby("patient_id").first()["y_norm"]
    c0 = per_pat.set_index("patient_id")["c0_calib"]
    r = (1.0 - y0 / c0).rename("r")
    r = r[c0 > 0]

    # A_0 from Trial C-damped calibration (per-patient corrected)
    A0 = v3d.set_index("patient_id")["A_0_used"]

    # Regime
    regime = per_pat.set_index("patient_id")["regime"]

    df = pd.DataFrame({"r": r, "A0": A0, "regime": regime}).dropna()
    return df


def main():
    print("Building cohort amplitude overview figure...")
    df = load_cohort_data()
    n_total = len(df)
    print(f"  Loaded {n_total} patients")
    print(f"  r quantiles: Q25={df['r'].quantile(0.25):.3f}, "
          f"median={df['r'].median():.3f}, Q75={df['r'].quantile(0.75):.3f}")
    print(f"  Thresholds: cap-feasibility = {THRESHOLD_CAP:.4f}, "
          f"global representable = {THRESHOLD_GLOBAL:.4f}")
    n_above_cap = int((df['r'] > THRESHOLD_CAP).sum())
    n_above_global = int((df['r'] > THRESHOLD_GLOBAL).sum())
    print(f"  r > cap-feasibility ({THRESHOLD_CAP:.3f}): {n_above_cap}/{n_total} "
          f"({100*n_above_cap/n_total:.1f}%)")
    print(f"  r > global representable ({THRESHOLD_GLOBAL:.3f}): {n_above_global}/{n_total} "
          f"({100*n_above_global/n_total:.1f}%)")

    fig = plt.figure(figsize=(13, 4.5))
    gs = fig.add_gridspec(1, 3, wspace=0.28, width_ratios=[1.1, 1.0, 1.4])

    # ------------------------------------------------------------------
    # Panel (a): Histogram of r_i, colour by regime, with thresholds
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0.4, 1.0, 25)
    bottom = np.zeros(len(bins) - 1)
    for regime in ["eradication", "dormancy", "escape"]:
        sub = df[df["regime"] == regime]["r"].values
        counts, _ = np.histogram(sub, bins=bins)
        ax.bar(0.5 * (bins[:-1] + bins[1:]), counts, width=np.diff(bins),
               bottom=bottom, color=REGIME_COLORS[regime], alpha=0.85,
               edgecolor="white", linewidth=0.4,
               label=f"{regime} (n={len(sub)})")
        bottom += counts

    # Threshold lines (vertical)
    ax.axvline(THRESHOLD_CAP, color="black", linestyle="--", lw=1.2,
               label=rf"cap-feasibility $A_0^{{\rm cap}} K_\alpha(1.5)\,L_F = {THRESHOLD_CAP:.3f}$")
    ax.axvline(THRESHOLD_GLOBAL, color="darkblue", linestyle=":", lw=1.4,
               label=rf"global representable $K_\alpha(1.5)/K_\alpha^{{\rm max}} = {THRESHOLD_GLOBAL:.3f}$")
    ax.set_xlabel(r"$r_i = 1 - y_{0,i}/c_{0,i}$  (data amplitude ratio)",
                  fontsize=10)
    ax.set_ylabel("patient count", fontsize=10)
    ax.set_title("(a) cohort distribution of $r_i$ vs amplitude bound thresholds",
                  fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.25, axis="y")

    # ------------------------------------------------------------------
    # Panel (b): A_0_i vs r_i scatter, with cap line
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for regime in ["eradication", "dormancy", "escape"]:
        sub = df[df["regime"] == regime]
        ax.scatter(sub["r"], sub["A0"], color=REGIME_COLORS[regime],
                    alpha=0.75, s=32, edgecolor="white", linewidth=0.4,
                    label=regime)
    # Theoretical relation: A_0 = r/(K_alpha(1.5) * L_F) (eq. A0_required)
    r_grid = np.linspace(0.0, 1.0, 100)
    A0_required = r_grid / (K_ALPHA_NOMINAL * L_F)
    ax.plot(r_grid, A0_required, color="black", linestyle="-", lw=1.0,
             label=r"$A_{0,i}^{\rm corr} = r_i/(K_\alpha(1.5)\,L_F)$")
    ax.axhline(A_0_CAP, color="darkred", linestyle="--", lw=1.2,
               label=rf"cap $A_0^{{\rm cap}} = {A_0_CAP:.1f}$")
    ax.set_xlabel(r"$r_i$  (data amplitude ratio)", fontsize=10)
    ax.set_ylabel(r"$A_{0,i}^{\rm corr}$  (calibrated kernel amplitude)",
                  fontsize=10)
    ax.set_title("(b) per-patient $A_0$ correction with contraction cap",
                  fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0.9, 2.3)

    # ------------------------------------------------------------------
    # Panel (c): per-patient ranked r_i with classification
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    df_sorted = df.sort_values("r", ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    x = np.arange(n)

    # Colour each bar by regime
    bar_colors = [REGIME_COLORS[reg] for reg in df_sorted["regime"]]
    ax.bar(x, df_sorted["r"], color=bar_colors, alpha=0.85,
           edgecolor="white", linewidth=0.3)

    # Threshold lines (horizontal)
    ax.axhline(THRESHOLD_CAP, color="black", linestyle="--", lw=1.2,
               label=rf"cap-feasibility $= {THRESHOLD_CAP:.3f}$")
    ax.axhline(THRESHOLD_GLOBAL, color="darkblue", linestyle=":", lw=1.4,
               label=rf"global representable $= {THRESHOLD_GLOBAL:.3f}$")

    # Annotate fractions
    n_above_cap_local = (df_sorted["r"] > THRESHOLD_CAP).sum()
    n_above_global_local = (df_sorted["r"] > THRESHOLD_GLOBAL).sum()
    ax.text(0.98, 0.97,
            f"$r_i > {THRESHOLD_CAP:.3f}$:  {n_above_cap_local}/{n} ({100*n_above_cap_local/n:.1f}%)\n"
            f"$r_i > {THRESHOLD_GLOBAL:.3f}$:  {n_above_global_local}/{n} ({100*n_above_global_local/n:.1f}%)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", lw=0.5))

    ax.set_xlabel("patient (ranked by descending $r_i$)", fontsize=10)
    ax.set_ylabel(r"$r_i$", fontsize=10)
    ax.set_title("(c) per-patient ranked $r_i$ vs framework bounds",
                  fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0.4, 1.02)
    ax.set_xlim(-0.5, n - 0.5)
    ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle(r"Bruchovsky OFF-phase cohort: data amplitude $r_i$ vs Theorem~9.1 "
                 r"representability bounds ($\alpha=1.5$, $p=3$, $L_F=1$)",
                 fontsize=11)

    out_pdf = figuras_dir() / "fig_S9_cohort_amplitude_overview.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_pdf}")


if __name__ == "__main__":
    main()
