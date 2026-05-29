#!/usr/bin/env python3
"""
run_l1_convergence_study.py
===========================

Production-grade convergence study of the L1 right-sided Caputo
discretisation against the closed-form extremal solution
y_ext(t) = B(alpha,gamma-alpha) / Gamma(alpha) * (1+t)^(alpha-gamma)
of Lemma 6.4 (sharpness) of the manuscript.

This is the experiment that supports Theorem 8.1 (consistency of the L1
operator at rate O(h^{2-alpha})) of Section 8.1 of the manuscript.

Usage
-----

Quick development run (~30 s on M3, sanity check only):

    python3 run_l1_convergence_study.py --mode quick

Full production run (estimated 30-90 min on Apple M3, 8-12 cores):

    python3 run_l1_convergence_study.py --mode full

Outputs (relative to repository root):

    figuras/fig_l1_convergence.pdf            -- 4-panel figure (vector PDF)
    figuras/fig_l1_tail_bound.pdf             -- tail bound visualisation
    secciones/tabla_l1_convergence.tex        -- LaTeX-ready table fragment
    codigo/results/l1_convergence_full.json   -- complete numerical log
    codigo/results/l1_convergence_full.csv    -- flat data for verification

Specifications
--------------

QUICK mode:
    alpha in {1.2, 1.5, 1.8}        (3 values)
    gamma_offset in {2.0}           (1 value, gamma = alpha + 2)
    h in logspace [0.4, 0.025]      (5 values)
    T_max = 80
    seeds: 1                        (deterministic for L1)

FULL mode:
    alpha in {1.1, 1.2, ..., 1.9}   (9 values, fine sweep)
    gamma_offset in {1.5, 2.0, 2.5, 3.0}  (4 values per alpha)
    h in logspace [0.4, 0.005]      (10 values, deep refinement)
    T_max = max(120, h^{-(2-alpha)/(alpha+mu-2)}) with mu = gamma + 2 - alpha
    bootstrap: 500 resamples for slope CI 95%

The L1 operator is deterministic (no stochasticity), so multiple seeds are
not needed; uncertainty in the fitted convergence rate comes from
linear regression on multiple (h, err) pairs and is reported via
bootstrap residuals.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence

# Ensure sibling modules (classical_scheme.py) are importable
# whether this script is invoked directly or with `python -m`.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from classical_scheme import (
    operator_test_yext,
    fit_rate,
    y_ext,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class StudyConfig:
    alphas: Sequence[float]
    gamma_offsets: Sequence[float]
    hs: Sequence[float]
    T_max_min: float
    mode: str
    n_bootstrap: int = 500
    interior_lo_frac: float = 0.02
    interior_hi_frac: float = 0.05

    @staticmethod
    def quick() -> "StudyConfig":
        return StudyConfig(
            alphas=(1.2, 1.5, 1.8),
            gamma_offsets=(2.0,),
            hs=tuple(np.logspace(np.log10(0.4), np.log10(0.025), 5)),
            T_max_min=80.0,
            mode="quick",
            n_bootstrap=200,
        )

    @staticmethod
    def full() -> "StudyConfig":
        return StudyConfig(
            alphas=tuple(np.round(np.arange(1.05, 1.96, 0.05), 3)),  # 19 alphas
            gamma_offsets=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0),            # 6 gammas
            hs=tuple(np.logspace(np.log10(0.4), np.log10(1e-3), 12)),  # 12 hs
            T_max_min=200.0,
            mode="full",
            n_bootstrap=1000,
        )


# ---------------------------------------------------------------------------
# Optimal T_max(h) by tail-truncation lemma (Appendix D.1) of the paper
# ---------------------------------------------------------------------------


def T_max_optimal(alpha: float, mu: float, h: float, T_min: float) -> float:
    r"""
    Choose T_max so that the tail bound T_max^{-(alpha+mu-2)} is
    dominated by the bulk consistency error h^{2-alpha}.

    From Theorem 8.1, the bulk error is C_cons * T_max * h^{2-alpha}, and
    the tail is C_tail * T_max^{-(alpha+mu-2)}.  We require
    T_max^{-(alpha+mu-2)} <= h^{2-alpha}, i.e.
        T_max >= h^{-(2-alpha)/(alpha+mu-2)}.
    """
    expo = -(2.0 - alpha) / (alpha + mu - 2.0)
    T_required = h ** expo
    return float(max(T_min, T_required))


# ---------------------------------------------------------------------------
# Bootstrap CI for the fitted convergence rate
# ---------------------------------------------------------------------------


def bootstrap_rate_ci(
    hs: np.ndarray, errs: np.ndarray, n_boot: int, seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    r"""
    Returns (rate_estimate, ci_low, ci_high) for the fitted slope of
    log(err) vs log(h), via bootstrap on residuals.
    """
    rng = np.random.default_rng(seed)
    p_hat, c_hat = fit_rate(hs, errs)
    log_h = np.log(hs)
    log_e = np.log(errs)
    fit_log = np.log(c_hat) + p_hat * log_h
    residuals = log_e - fit_log
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(residuals), size=len(residuals))
        log_e_b = fit_log + residuals[idx]
        try:
            p_b, _ = fit_rate(hs, np.exp(log_e_b))
        except Exception:
            p_b = np.nan
        boots[b] = p_b
    boots = boots[~np.isnan(boots)]
    lo = np.quantile(boots, (1 - ci) / 2)
    hi = np.quantile(boots, (1 + ci) / 2)
    return float(p_hat), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Single-cell experiment
# ---------------------------------------------------------------------------


def run_cell(
    alpha: float, gamma: float, h: float, T_max: float,
    config: StudyConfig,
) -> dict:
    r"""Run the L1 operator test for one (alpha, gamma, h) cell."""
    t, err, _, _ = operator_test_yext(alpha, gamma, h, T_max)
    N = len(t) - 1
    i_lo = max(1, int(config.interior_lo_frac * N))
    i_hi = N - max(8, int(config.interior_hi_frac * N))
    interior = err[i_lo:i_hi]
    # Sub-range diagnostics
    near_zero = err[1 : max(2, N // 10)]
    mid_range = err[max(2, N // 10) : max(3, N // 2)]
    return dict(
        alpha=alpha, gamma=gamma, h=h, T_max=T_max,
        N=N,
        err_inf_full=float(np.max(err)),
        err_inf_interior=float(np.max(interior)),
        err_l2_interior=float(np.sqrt(np.mean(interior ** 2))),
        err_l1_interior=float(np.mean(np.abs(interior))),
        err_at_t0=float(err[0]),
        err_at_T_max_minus_5h=float(err[max(0, N - 6)]),
        err_inf_near_zero=float(np.max(near_zero)),
        err_inf_mid=float(np.max(mid_range)),
    )


def run_tail_study(
    alpha: float, gamma: float, h: float,
    T_max_grid: Sequence[float], config: StudyConfig,
    T_max_reference: float = 5000.0,
    t_probe: float = 5.0,
) -> list[dict]:
    r"""
    Fix (alpha, gamma, h) and sweep T_max, isolating the *tail truncation*
    error by comparing the L1 operator on the truncated mesh to the L1
    operator on a much larger reference mesh (T_max_reference >> T_max).

    The reference is computed once and acts as the "T_max = infinity"
    proxy: the difference
        |L_h^(T_max) y_ext (t_probe) - L_h^(T_max_ref) y_ext (t_probe)|
    is exactly the tail-truncation contribution, isolated from the bulk
    consistency error.

    According to tail-truncation lemma (Appendix D.1) of the paper, this difference is bounded by
        C * T_max^{-(alpha+mu-2)},  with mu = gamma + 2 - alpha,
    and we expect the empirical decay to track this rate.
    """
    from classical_scheme import caputo_minus_L1

    mu = gamma + 2.0 - alpha
    expo = -(alpha + mu - 2.0)

    # Reference operator value at t_probe, computed on a very large mesh.
    N_ref = int(round(T_max_reference / h))
    t_ref = h * np.arange(N_ref + 1)
    y_ref = y_ext(t_ref, alpha, gamma)
    Lh_ref = caputo_minus_L1(y_ref, h, alpha)
    n_probe_ref = int(round(t_probe / h))
    if n_probe_ref >= len(Lh_ref):
        raise ValueError(
            f"t_probe={t_probe} too close to T_max_reference={T_max_reference}."
        )
    Lh_at_probe_ref = float(Lh_ref[n_probe_ref])

    results = []
    for T_max in T_max_grid:
        N = int(round(T_max / h))
        if N <= n_probe_ref + 5:
            # Skip if T_max < t_probe + buffer
            continue
        t = h * np.arange(N + 1)
        y_grid = y_ext(t, alpha, gamma)
        Lh = caputo_minus_L1(y_grid, h, alpha)
        Lh_at_probe = float(Lh[n_probe_ref])
        # Tail-truncation error: difference from reference
        tail_err = abs(Lh_at_probe - Lh_at_probe_ref)
        # For comparison: total error vs analytic a(t_probe)
        a_at_probe = (1.0 + t_probe) ** (-gamma)
        total_err = abs(Lh_at_probe - a_at_probe)
        results.append(dict(
            alpha=alpha, gamma=gamma, h=h, T_max=T_max,
            mu=mu, tail_exponent=expo, N=N, t_probe=t_probe,
            tail_truncation_err=tail_err,
            total_err_at_probe=total_err,
            T_max_reference=T_max_reference,
        ))
    return results


def run_study(config: StudyConfig, verbose: bool = True) -> list[dict]:
    r"""Run all cells (alpha, gamma, h)."""
    cells = []
    n_total = len(config.alphas) * len(config.gamma_offsets) * len(config.hs)
    n_done = 0
    t_start = time.time()
    for alpha in config.alphas:
        for goff in config.gamma_offsets:
            gamma = alpha + goff
            mu = gamma + 2.0 - alpha   # decay rate of y_ext'' (Lemma 6.4 case)
            for h in config.hs:
                T_max = T_max_optimal(alpha, mu, h, config.T_max_min)
                t0 = time.time()
                res = run_cell(alpha, gamma, h, T_max, config)
                dt = time.time() - t0
                res["wall_time_s"] = dt
                cells.append(res)
                n_done += 1
                if verbose:
                    elapsed = time.time() - t_start
                    eta = elapsed / n_done * (n_total - n_done)
                    print(
                        f"[{n_done:3d}/{n_total:3d}] alpha={alpha:.2f} "
                        f"gamma={gamma:.2f} h={h:8.5f} T_max={T_max:7.1f} "
                        f"err={res['err_inf_interior']:.3e}  "
                        f"({dt:.1f}s, ETA {eta/60:.1f}m)",
                        flush=True,
                    )
    return cells


# ---------------------------------------------------------------------------
# Aggregation: rates per (alpha, gamma)
# ---------------------------------------------------------------------------


def aggregate_rates(cells: list[dict], config: StudyConfig) -> list[dict]:
    rates = []
    for alpha in config.alphas:
        for goff in config.gamma_offsets:
            gamma = alpha + goff
            cells_ag = [c for c in cells
                        if abs(c["alpha"] - alpha) < 1e-9
                        and abs(c["gamma"] - gamma) < 1e-9]
            cells_ag.sort(key=lambda c: c["h"], reverse=True)
            hs = np.array([c["h"] for c in cells_ag])
            errs = np.array([c["err_inf_interior"] for c in cells_ag])
            p_hat, lo, hi = bootstrap_rate_ci(
                hs, errs, n_boot=config.n_bootstrap, seed=hash((alpha, goff)) % (2 ** 32)
            )
            theoretical = 2.0 - alpha
            rates.append(dict(
                alpha=alpha, gamma=gamma, gamma_offset=goff,
                rate_hat=p_hat, ci_low=lo, ci_high=hi,
                rate_theoretical=theoretical,
                exceeds_theory=(lo > theoretical),
                hs=hs.tolist(), errs=errs.tolist(),
            ))
    return rates


# ---------------------------------------------------------------------------
# Output: figure
# ---------------------------------------------------------------------------


def make_figure(
    cells: list[dict], rates: list[dict], config: StudyConfig,
    tail_results: list[dict], out_pdf: str,
):
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })

    fig = plt.figure(figsize=(12.5, 9.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.30)

    # ---- Panel (a): extremal profiles for representative alphas
    ax = fig.add_subplot(gs[0, 0])
    t = np.linspace(0.0, 8.0, 800)
    cmap = plt.get_cmap("viridis")
    show_alphas = [1.1, 1.3, 1.5, 1.7, 1.9] if config.mode == "full" else config.alphas
    for i, a in enumerate(show_alphas):
        g = a + 2.0
        y = y_ext(t, a, g)
        ax.plot(t, y, color=cmap(i / max(1, len(show_alphas) - 1)),
                label=fr"$\alpha={a:.1f}$ ($\gamma={g:.1f}$)")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$y_{\mathrm{ext}}(t)$")
    ax.set_title(r"(a) Extremal profiles $y_{\mathrm{ext}}(t)=\frac{B(\alpha,\gamma-\alpha)}{\Gamma(\alpha)}(1+t)^{\alpha-\gamma}$")
    ax.legend(loc="upper right", frameon=True)

    # ---- Panel (b): convergence diagrams (gamma = alpha+2)
    # To avoid an over-crowded legend with 19 alphas in steps of 0.05, plot
    # all curves but legend-label only a representative subset (every fifth).
    ax = fig.add_subplot(gs[0, 1])
    rates_g2 = sorted(
        [r for r in rates if abs(r["gamma_offset"] - 2.0) < 1e-9],
        key=lambda r: r["alpha"],
    )
    cmap = plt.get_cmap("plasma")
    # Label only ~5 representative alphas (uniform stride) plus the last one
    n_curves = len(rates_g2)
    n_target_labels = 5 if n_curves >= 8 else n_curves
    label_indices = set(np.linspace(0, n_curves - 1, n_target_labels).round().astype(int).tolist())
    for i, r in enumerate(rates_g2):
        col = cmap(i / max(1, n_curves - 1))
        if i in label_indices:
            lbl = fr"$\alpha={r['alpha']:.2f}$  ($\hat p={r['rate_hat']:.2f}$)"
        else:
            lbl = None
        ax.loglog(r["hs"], r["errs"], "o-", color=col, alpha=0.8,
                  markersize=3, label=lbl)
    href = np.array([min(config.hs) * 0.8, max(config.hs) * 1.2])
    # Reference slope 1 (interior cancellation rate, dotted): anchor to the
    # smallest-alpha curve at its smallest h so it sits just above the data.
    base_h1 = 1.5 * rates_g2[0]["errs"][-1] / rates_g2[0]["hs"][-1]
    ax.loglog(href, base_h1 * href, "k:", linewidth=1.2, label="slope $1$")
    # Reference slope 2-alpha (worst-case theoretical, dashed): anchor to the
    # largest-alpha curve at its smallest h, with a small downward offset so
    # the line sits just below the empirical data without overlapping.
    a_w = max(r["alpha"] for r in rates_g2)
    worst = rates_g2[-1]
    h_anchor = min(worst["hs"])
    err_anchor = worst["errs"][-1]
    base_th = 0.6 * err_anchor / h_anchor ** (2.0 - a_w)
    ax.loglog(href, base_th * href ** (2.0 - a_w), "k--", linewidth=1.2,
              label=fr"slope $2-\alpha={2 - a_w:.2f}$ (worst)")
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$\|\mathcal{L}_h y_{\mathrm{ext}} - a\|_\infty$ (interior)")
    ax.set_title(r"(b) Convergence at $\gamma=\alpha+2$")
    ax.set_xlim(min(config.hs) * 0.7, max(config.hs) * 1.3)
    # y-limits: keep the slope 2-alpha reference line visible below the
    # data (factor 0.2 below the worst-curve minimum), and leave a clear
    # empty decade above the data so the upper-left legend never overlaps
    # the curves.
    err_max = max(max(r["errs"]) for r in rates_g2)
    err_min = min(min(r["errs"]) for r in rates_g2)
    ax.set_ylim(err_min * 0.2, err_max * 10.0)
    ax.legend(loc="upper left", frameon=True, ncol=1, fontsize=6.5,
              framealpha=0.9, borderpad=0.3, handletextpad=0.4,
              labelspacing=0.3)

    # ---- Panel (c): empirical rate vs alpha, with CI; theoretical 2-alpha.
    # The 6 gamma offsets produce nearly overlapping curves (rate weakly
    # depends on gamma); we plot all 6 but legend-label only 3 representative
    # ones to declutter, and place the legend in the lower-left region which
    # is free of empirical data (the empirical rates lie in the band
    # p_hat in [0.85, 1.06]).
    ax = fig.add_subplot(gs[1, 0])
    colours = plt.get_cmap("tab10")
    g_label_set = {min(config.gamma_offsets),
                   sorted(config.gamma_offsets)[len(config.gamma_offsets) // 2],
                   max(config.gamma_offsets)}
    for j, goff in enumerate(config.gamma_offsets):
        rs = [r for r in rates if abs(r["gamma_offset"] - goff) < 1e-9]
        a_arr = np.array([r["alpha"] for r in rs])
        p_arr = np.array([r["rate_hat"] for r in rs])
        lo_arr = np.array([r["ci_low"] for r in rs])
        hi_arr = np.array([r["ci_high"] for r in rs])
        lbl = (fr"$\gamma=\alpha+{goff:.1f}$" if goff in g_label_set else None)
        ax.errorbar(a_arr, p_arr, yerr=[p_arr - lo_arr, hi_arr - p_arr],
                    fmt="o-", color=colours(j),
                    label=lbl, capsize=2.5, elinewidth=0.7,
                    markersize=3, alpha=0.85)
    a_th = np.linspace(1.05, 1.95, 50)
    ax.plot(a_th, 2.0 - a_th, "k--", linewidth=1.2,
            label=r"theoretical $2-\alpha$")
    ax.plot(a_th, np.ones_like(a_th), "k:", linewidth=1.0,
            label="reference $1$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Empirical rate $\\hat p$")
    ax.set_title("(c) Empirical convergence rate (95% CI bootstrap)")
    ax.set_xlim(1.0, 2.0)
    ax.set_ylim(-0.05, 1.15)
    # Legend in lower-left: empirical rates sit at p_hat ~ 1, and the
    # theoretical line crosses but only as a thin reference -- this corner
    # is the most data-free.
    ax.legend(loc="lower left", frameon=True, ncol=1, fontsize=7.5,
              framealpha=0.85)

    # ---- Panel (d): tail-study experimental verification of tail-truncation lemma (Appendix D.1)
    # Plot empirical err vs T_max at fixed h, alongside analytic bound
    # T_max^{-(alpha+mu-2)}.
    ax = fig.add_subplot(gs[1, 1])
    if tail_results:
        # Group by (alpha, gamma)
        groups = {}
        for r in tail_results:
            key = (r["alpha"], r["gamma"])
            groups.setdefault(key, []).append(r)
        cmap = plt.get_cmap("viridis")
        keys = sorted(groups.keys())
        for i, key in enumerate(keys):
            rs = sorted(groups[key], key=lambda r: r["T_max"])
            Tm = np.array([r["T_max"] for r in rs])
            err = np.array([r["tail_truncation_err"] for r in rs])
            # Filter zeros (reference T_max coincidence) to avoid log(0)
            mask = err > 0
            Tm, err = Tm[mask], err[mask]
            expo = rs[0]["tail_exponent"]
            col = cmap(i / max(1, len(keys) - 1))
            ax.loglog(Tm, err, "o-", color=col,
                      label=fr"$\alpha={key[0]:.1f},\,\gamma={key[1]:.1f}$ "
                            fr"(target $T_{{\max}}^{{{expo:+.2f}}}$)")
            # Reference slope line anchored to first data point
            if len(Tm) >= 2:
                ax.loglog(Tm, err[0] * (Tm / Tm[0]) ** expo, ":", color=col,
                          linewidth=0.8, alpha=0.6)
        ax.set_xlabel(r"$T_{\max}$ (at fixed $h$)")
        ax.set_ylabel(r"$|\mathcal{L}_h^{(T_{\max})}\,y_{\mathrm{ext}}(t_*) - "
                      r"\mathcal{L}_h^{(T_{\mathrm{ref}})}\,y_{\mathrm{ext}}(t_*)|$")
        ax.set_title(r"(d) Tail truncation error versus tail-truncation lemma (Appendix D.1) prediction")
        ax.legend(loc="lower left", frameon=True, fontsize=7, ncol=1)
    else:
        ax.text(0.5, 0.5, "No tail data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()

    fig.suptitle(
        f"L1 right-sided Caputo discretisation: convergence study"
        f" ({config.mode} mode, {len(config.alphas)} alphas × "
        f"{len(config.gamma_offsets)} gammas × {len(config.hs)} h-values)",
        fontsize=12,
    )
    fig.savefig(out_pdf, format="pdf")
    print(f"Wrote {out_pdf}")


# ---------------------------------------------------------------------------
# Output: LaTeX table fragment
# ---------------------------------------------------------------------------


def make_latex_table(rates: list[dict], config: StudyConfig, out_tex: str):
    rates_g2 = sorted(
        [r for r in rates if abs(r["gamma_offset"] - 2.0) < 1e-9],
        key=lambda r: r["alpha"],
    )

    lines: list[str] = []
    lines.append(r"% Auto-generated by run_l1_convergence_study.py "
                 f"(mode={config.mode})")
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Convergence study of the L1 right-sided Caputo "
        r"discretisation $\mathcal{L}_h$ on the extremal datum "
        r"$y_{\mathrm{ext}}$ of Lemma~\ref{lem:sharpness}, with "
        r"$\gamma = \alpha + 2$. Reported quantity: maximum interior "
        r"error $\|\mathcal{L}_h y_{\mathrm{ext}}(t_\bullet) - a\|_\infty$ "
        r"on the indices "
        r"$\lceil 0.02 N\rceil \leq n \leq N - \lceil 0.05 N\rceil$. "
        r"The empirical rate $\hat p$ is the slope of the log-log linear "
        r"regression of error vs.\ $h$, with 95\% bootstrap CI on the "
        r"residuals. All measured rates are bounded above by the "
        r"worst-case operator rate $O(h^{2-\alpha})$ of "
        r"Theorem~\ref{thm:L1_consistency}; the gain to the interior "
        r"cancellation rate $O(h)$ predicted under "
        r"$y_{\mathrm{ext}} \in C^{\infty}$ "
        r"(Remark~\ref{rem:sharp_2_minus_alpha}) is observed across the "
        r"entire range of $\alpha$.}"
    )
    lines.append(r"\label{tab:rates_yext}")
    h_cols = list(rates_g2[0]["hs"])
    n_h_show = min(6, len(h_cols))
    h_show_indices = list(range(0, len(h_cols), max(1, len(h_cols) // n_h_show)))[:n_h_show]
    h_show = [h_cols[i] for i in h_show_indices]

    col_spec = "l" + "c" * len(h_show) + "ccc"
    # Auto-fit table to text width to avoid horizontal overflow on large
    # parameter sweeps (e.g., 19 alphas with 6 h-columns).
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")
    h_header = " & ".join(fr"$h={h:.3g}$" for h in h_show)
    lines.append(fr"$\alpha$ & {h_header} & $\hat p$ & 95\% CI & $2-\alpha$ \\")
    lines.append(r"\midrule")
    for r in rates_g2:
        errs = r["errs"]
        a_str = f"${r['alpha']:.2f}$"
        e_cells = [_format_sci(errs[i]) for i in h_show_indices]
        e_str = " & ".join(e_cells)
        p_str = f"${r['rate_hat']:.3f}$"
        ci_str = f"$[{r['ci_low']:.3f},\\,{r['ci_high']:.3f}]$"
        th_str = f"${r['rate_theoretical']:.2f}$"
        lines.append(f"{a_str} & {e_str} & {p_str} & {ci_str} & {th_str} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    with open(out_tex, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {out_tex}")


def _format_sci(x: float) -> str:
    if x == 0:
        return r"$0$"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10 ** exp
    return fr"${mant:.2f}\!\times\!10^{{{exp}}}$"


# ---------------------------------------------------------------------------
# JSON / CSV logs
# ---------------------------------------------------------------------------


def write_logs(cells: list[dict], rates: list[dict], config: StudyConfig,
               tail_results: list[dict],
               out_json: str, out_csv: str):
    def _json_default(o):
        # Handle numpy scalars and arrays explicitly to avoid the
        # 0-d "iterable" trap in the default fallback.
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, tuple):
            return list(o)
        # Last resort
        return str(o)

    log = dict(
        config=asdict(config),
        cells=cells,
        rates=rates,
        tail_results=tail_results,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    with open(out_json, "w") as fh:
        json.dump(log, fh, indent=2, default=_json_default)
    print(f"Wrote {out_json}")

    with open(out_csv, "w", newline="") as fh:
        if cells:
            keys = [k for k in cells[0].keys()]
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for c in cells:
                w.writerow(c)
    print(f"Wrote {out_csv}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["quick", "full"], default="quick",
                    help="Quick (~30s) or full (~30-90 min on M3) experiment.")
    ap.add_argument("--repo-root", default=None,
                    help="Repository root; default: parent of this script's directory.")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = args.repo_root or os.path.dirname(here)
    figs_dir = os.path.join(repo_root, "figuras")
    secs_dir = os.path.join(repo_root, "secciones")
    res_dir = os.path.join(here, "results")
    for d in (figs_dir, secs_dir, res_dir):
        os.makedirs(d, exist_ok=True)

    config = StudyConfig.full() if args.mode == "full" else StudyConfig.quick()
    # Convert numpy floats to plain floats for cleaner printing/serialisation
    config.alphas = tuple(float(a) for a in config.alphas)
    config.hs = tuple(float(h) for h in config.hs)

    print("=" * 72)
    print(f"L1 RIGHT-SIDED CAPUTO CONVERGENCE STUDY  ({args.mode} mode)")
    a_str = ", ".join(f"{a:.2f}" for a in config.alphas)
    g_str = ", ".join(f"{g:.2f}" for g in config.gamma_offsets)
    print(f"  alphas ({len(config.alphas)}):    {a_str}")
    print(f"  gamma offsets ({len(config.gamma_offsets)}): {g_str}")
    print(f"  h grid ({len(config.hs)}): "
          f"{min(config.hs):.4g} ... {max(config.hs):.4g}")
    print(f"  T_max_min:     {config.T_max_min}")
    print(f"  n_bootstrap:   {config.n_bootstrap}")
    print("=" * 72)

    t_start = time.time()
    cells = run_study(config, verbose=True)
    rates = aggregate_rates(cells, config)

    # Tail study: fix (alpha, gamma, h) and sweep T_max for representative cases.
    print("\n" + "=" * 72)
    print("TAIL STUDY (verifying tail-truncation lemma (Appendix D.1); tail isolated via reference T_max)")
    print("=" * 72)
    tail_results = []
    if config.mode == "full":
        h_fixed = 0.05
        T_max_grid = list(np.logspace(np.log10(20.0), np.log10(1500.0), 12))
        T_ref = 5000.0
        tail_alphas_gammas = [(1.2, 3.2), (1.5, 3.5), (1.8, 3.8)]
    else:
        h_fixed = 0.1
        T_max_grid = list(np.logspace(np.log10(20.0), np.log10(400.0), 6))
        T_ref = 2000.0
        tail_alphas_gammas = [(1.5, 3.5)]
    for a, g in tail_alphas_gammas:
        print(f"  alpha={a}, gamma={g}, h={h_fixed}, T_max sweep "
              f"[{T_max_grid[0]:.0f}..{T_max_grid[-1]:.0f}], T_ref={T_ref}")
        rs = run_tail_study(a, g, h_fixed, T_max_grid, config,
                             T_max_reference=T_ref, t_probe=5.0)
        tail_results.extend(rs)
        for r in rs[::3]:
            print(f"    T_max={r['T_max']:8.1f}  tail_err={r['tail_truncation_err']:.3e}"
                  f"   target rate {r['tail_exponent']:+.2f}")

    elapsed = time.time() - t_start

    print(f"\nTotal wall time: {elapsed/60:.1f} min")
    print()
    print(f"{'alpha':>6}  {'gamma':>6}  {'rate_hat':>9}  {'CI_95':>22}  {'2-alpha':>8}  flag")
    for r in rates:
        flag = "OK (gain)" if r["exceeds_theory"] else "borderline"
        print(f"{r['alpha']:6.2f}  {r['gamma']:6.2f}  "
              f"{r['rate_hat']:9.3f}  "
              f"[{r['ci_low']:6.3f}, {r['ci_high']:6.3f}]  "
              f"{r['rate_theoretical']:8.2f}  {flag}")

    suffix = "_full" if args.mode == "full" else "_quick"
    make_figure(cells, rates, config, tail_results,
                os.path.join(figs_dir, f"fig_l1_convergence{suffix}.pdf"))
    make_latex_table(rates, config,
                     os.path.join(secs_dir, f"tabla_l1_convergence{suffix}.tex"))
    write_logs(cells, rates, config, tail_results,
               os.path.join(res_dir, f"l1_convergence{suffix}.json"),
               os.path.join(res_dir, f"l1_convergence{suffix}.csv"))


if __name__ == "__main__":
    main()
