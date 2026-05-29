"""
benchmark_hba_paired_inference.py
==================================

Paired-inference layer for the HBA-2010 vs §7 benchmark.

Adds, to the point estimates already reported in benchmark_hba_exact_stats.txt:

  (i)   Bootstrap 95% CI for the median per-patient test RMSE ratio HBA/§7.
  (ii)  Wilcoxon signed-rank paired test on train and test RMSE.
  (iii) Bootstrap 95% CI for the cohort-wide failure-rate ratio (HBA/§7).
  (iv)  Sensitivity of the failure-rate ratio to the threshold
        tau in {0.3, 0.5, 0.7, 1.0}.

Reproducibility: numpy default_rng with explicit seed (BOOTSTRAP_SEED).

Inputs:
    outputs_clinical/benchmark_hba_exact.csv
    outputs_clinical/bruchovsky_calibration_offphase_v3_damped.csv

Outputs:
    outputs_clinical/benchmark_hba_paired_inference.txt
    outputs_clinical/benchmark_hba_paired_inference_table.tex
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2].parent
OUT = ROOT / "outputs_clinical"
HBA_CSV = OUT / "benchmark_hba_exact.csv"
S7_CSV = OUT / "bruchovsky_calibration_offphase_v3_damped.csv"
TXT_OUT = OUT / "benchmark_hba_paired_inference.txt"
TEX_OUT = OUT / "benchmark_hba_paired_inference_table.tex"

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20260512
ALPHA_CI = 0.05

COHORT_N = 55  # total patients in the cohort
TAU_GRID = (0.3, 0.5, 0.7, 1.0)
TAU_PAPER = 0.5  # the threshold reported in the paper


def bootstrap_median_ratio(
    num: np.ndarray, den: np.ndarray, B: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Paired bootstrap of median(num/den). Returns (point, lo, hi)."""
    assert num.shape == den.shape
    ratios = num / den
    point = float(np.median(ratios))
    n = len(ratios)
    if n == 0:
        return point, float("nan"), float("nan")
    idx = rng.integers(0, n, size=(B, n))
    boot = np.median(ratios[idx], axis=1)
    lo, hi = np.quantile(boot, [ALPHA_CI / 2, 1 - ALPHA_CI / 2])
    return point, float(lo), float(hi)


def bootstrap_proportion(x: np.ndarray, B: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Bootstrap CI for a proportion. x is a 0/1 array. Returns (p, lo, hi)."""
    n = len(x)
    p = float(np.mean(x))
    idx = rng.integers(0, n, size=(B, n))
    boot = np.mean(x[idx], axis=1)
    lo, hi = np.quantile(boot, [ALPHA_CI / 2, 1 - ALPHA_CI / 2])
    return p, float(lo), float(hi)


def bootstrap_proportion_ratio(
    a: np.ndarray, b: np.ndarray, B: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Bootstrap CI for ratio of two proportions (paired by patient: a, b in {0,1})."""
    n = len(a)
    assert b.shape == a.shape
    p_a = float(np.mean(a))
    p_b = float(np.mean(b))
    point = p_a / p_b if p_b > 0 else float("inf")
    idx = rng.integers(0, n, size=(B, n))
    pa_boot = np.mean(a[idx], axis=1)
    pb_boot = np.mean(b[idx], axis=1)
    mask = pb_boot > 0
    ratio = pa_boot[mask] / pb_boot[mask]
    if len(ratio) < 100:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(ratio, [ALPHA_CI / 2, 1 - ALPHA_CI / 2])
    return point, float(lo), float(hi)


def fmt(x: float, d: int = 4) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.{d}f}"


def fmt_pvalue_tex(p: float) -> str:
    """Format a p-value in scientific notation suitable for LaTeX math mode."""
    if p == 0.0:
        return r"<\,10^{-16}"
    exp = int(np.floor(np.log10(p)))
    mant = p / (10**exp)
    return f"{mant:.1f}\\times 10^{{{exp}}}"


def fmt_ratio_tex(r: float, lo: float, hi: float) -> str:
    """Format ratio with CI for LaTeX math mode; handles inf/nan gracefully."""
    if not np.isfinite(r):
        return r"+\infty\;[\,\text{---}\,]"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{r:.2f}\\;[\\,\\text{{---}}\\,]"
    return f"{r:.2f}\\;[{lo:.2f},\\,{hi:.2f}]"


def fmt_prop_tex(p: float, lo: float, hi: float) -> str:
    return f"{p:.3f}\\;[{lo:.3f},\\,{hi:.3f}]"


def main() -> None:
    hba = pd.read_csv(HBA_CSV)
    s7 = pd.read_csv(S7_CSV)

    # Keep only converged patients in each method.
    hba_ok = hba[hba["status"] == "ok"].copy()
    s7_ok = s7[s7["status"] == "ok"].copy()

    merged = pd.merge(
        hba_ok[["patient_id", "regime", "train_RMSE", "test_RMSE"]],
        s7_ok[["patient_id", "train_RMSE", "test_RMSE"]],
        on="patient_id",
        suffixes=("_hba", "_s7"),
    )
    N_paired = len(merged)

    # ---------- (i) Median per-patient test RMSE ratio + CI ----------
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ratio_test = merged["test_RMSE_hba"].to_numpy() / merged["test_RMSE_s7"].to_numpy()
    med_test, lo_test, hi_test = bootstrap_median_ratio(
        merged["test_RMSE_hba"].to_numpy(),
        merged["test_RMSE_s7"].to_numpy(),
        BOOTSTRAP_B,
        rng,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    med_train, lo_train, hi_train = bootstrap_median_ratio(
        merged["train_RMSE_hba"].to_numpy(),
        merged["train_RMSE_s7"].to_numpy(),
        BOOTSTRAP_B,
        rng,
    )

    # ---------- (ii) Wilcoxon signed-rank paired tests ----------
    w_train = wilcoxon(
        merged["train_RMSE_hba"].to_numpy(),
        merged["train_RMSE_s7"].to_numpy(),
        alternative="two-sided",
        zero_method="wilcox",
    )
    w_test = wilcoxon(
        merged["test_RMSE_hba"].to_numpy(),
        merged["test_RMSE_s7"].to_numpy(),
        alternative="two-sided",
        zero_method="wilcox",
    )

    # ---------- (iii) Cohort-wide failure-rate ratio ----------
    # Failure = non-converged OR test_RMSE > tau. To compute cohort-wide we need
    # to assemble the full N=55 list, marking unconverged patients as "failure" for
    # the method that did not converge.
    # Build a single patient_id index that covers union of all patients seen.
    all_ids = sorted(set(hba["patient_id"]).union(s7["patient_id"]))
    assert len(all_ids) == COHORT_N, f"Expected {COHORT_N} unique patients, got {len(all_ids)}"

    df_full = pd.DataFrame({"patient_id": all_ids})
    hba_idx = hba.set_index("patient_id")
    s7_idx = s7.set_index("patient_id")

    def method_fail(method_df, ids, tau):
        out = np.zeros(len(ids), dtype=int)
        for k, pid in enumerate(ids):
            row = method_df.loc[pid] if pid in method_df.index else None
            if row is None or row["status"] != "ok":
                out[k] = 1
            elif row["test_RMSE"] > tau:
                out[k] = 1
        return out

    # Sensitivity over tau ----------------------------------------------------
    sens_rows = []
    for tau in TAU_GRID:
        fail_hba = method_fail(hba_idx, all_ids, tau)
        fail_s7 = method_fail(s7_idx, all_ids, tau)
        rng = np.random.default_rng(BOOTSTRAP_SEED + 100 + int(tau * 10))
        p_hba, lo_hba, hi_hba = bootstrap_proportion(fail_hba, BOOTSTRAP_B, rng)
        rng = np.random.default_rng(BOOTSTRAP_SEED + 200 + int(tau * 10))
        p_s7, lo_s7, hi_s7 = bootstrap_proportion(fail_s7, BOOTSTRAP_B, rng)
        rng = np.random.default_rng(BOOTSTRAP_SEED + 300 + int(tau * 10))
        point_ratio, lo_ratio, hi_ratio = bootstrap_proportion_ratio(
            fail_hba, fail_s7, BOOTSTRAP_B, rng
        )
        sens_rows.append(
            dict(
                tau=tau,
                p_hba=p_hba,
                lo_hba=lo_hba,
                hi_hba=hi_hba,
                p_s7=p_s7,
                lo_s7=lo_s7,
                hi_s7=hi_s7,
                ratio=point_ratio,
                lo_ratio=lo_ratio,
                hi_ratio=hi_ratio,
            )
        )
    sens = pd.DataFrame(sens_rows)

    # ---------- Write text summary ----------
    with TXT_OUT.open("w") as fh:
        fh.write("HBA-2010 vs §7 Trial C-damped — paired inference layer\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(f"Bootstrap B = {BOOTSTRAP_B}, seed = {BOOTSTRAP_SEED}, "
                 f"alpha = {ALPHA_CI} (95% CI)\n")
        fh.write(f"Patients (both converged): N_paired = {N_paired}\n")
        fh.write(f"Cohort total: N = {COHORT_N}\n\n")

        fh.write("(i) Median per-patient RMSE ratio HBA/§7 with paired bootstrap CI\n")
        fh.write("-" * 60 + "\n")
        fh.write(f"  Train:  median = {fmt(med_train)}  "
                 f"[95% CI: {fmt(lo_train)}, {fmt(hi_train)}]\n")
        fh.write(f"  Test:   median = {fmt(med_test)}  "
                 f"[95% CI: {fmt(lo_test)}, {fmt(hi_test)}]\n\n")

        fh.write("(ii) Wilcoxon signed-rank paired test (two-sided)\n")
        fh.write("-" * 60 + "\n")
        fh.write(f"  Train RMSE:  W = {w_train.statistic:.2f},  p = {w_train.pvalue:.3e}\n")
        fh.write(f"  Test  RMSE:  W = {w_test.statistic:.2f},  p = {w_test.pvalue:.3e}\n\n")

        fh.write("(iii) Cohort-wide failure-rate sensitivity to threshold tau\n")
        fh.write("      failure = non-convergence OR test_RMSE > tau\n")
        fh.write("-" * 60 + "\n")
        fh.write(
            f"{'tau':>6} | {'p_HBA':>16} | {'p_§7':>16} | "
            f"{'ratio HBA/§7':>22}\n"
        )
        fh.write("-" * 60 + "\n")
        for _, r in sens.iterrows():
            fh.write(
                f"{r.tau:>6.2f} | "
                f"{r.p_hba:>5.3f} [{r.lo_hba:.3f},{r.hi_hba:.3f}] | "
                f"{r.p_s7:>5.3f} [{r.lo_s7:.3f},{r.hi_s7:.3f}] | "
                f"{r.ratio:>5.2f} [{r.lo_ratio:.2f},{r.hi_ratio:.2f}]\n"
            )
        fh.write("\nNote: tau = 0.5 is the threshold reported in the paper.\n")

    # ---------- Write LaTeX table for §9.3 (HBA benchmark) ----------
    with TEX_OUT.open("w") as fh:
        fh.write(
            "% Auto-generated by Github/src/benchmark/benchmark_hba_paired_inference.py\n"
        )
        fh.write("% B = " + str(BOOTSTRAP_B) + ", seed = " + str(BOOTSTRAP_SEED) + ", N_paired = " + str(N_paired) + "\n")
        fh.write("\\begin{table}[t]\n")
        fh.write("\\centering\n")
        fh.write("\\small\n")
        fh.write(
            "\\caption{Paired inference for the HBA-2010 vs.\\ \\S\\ref{sec:tumor_inmune} "
            "benchmark on the Bruchovsky $N=55$ IADT cohort.\n"
            "Top block: median per-patient RMSE ratio $\\mathrm{HBA}/\\S\\ref{sec:tumor_inmune}$ with "
            "paired bootstrap $95\\%$ confidence intervals ($B = " + str(BOOTSTRAP_B) + "$, "
            "$N_{\\mathrm{paired}} = " + str(N_paired) + "$) and Wilcoxon "
            "signed-rank $p$-values.\n"
            "Bottom block: sensitivity of the cohort-wide failure-rate ratio "
            "(failure $=$ non-convergence or test RMSE $> \\tau$, $N = " + str(COHORT_N) + "$) "
            "to the threshold $\\tau$; $\\tau = 0.5$ is the value reported in the body.}\n"
        )
        fh.write("\\label{tab:hba_paired_inference}\n")
        fh.write("\\begin{tabular}{lccc}\n")
        fh.write("\\toprule\n")
        fh.write(
            "Quantity & Point estimate & 95\\% CI & Wilcoxon $p$ \\\\\n"
        )
        fh.write("\\midrule\n")
        fh.write(
            f"Median train-RMSE ratio HBA/\\S\\ref{{sec:tumor_inmune}} & "
            f"${med_train:.3f}$ & "
            f"$[{lo_train:.3f},\\,{hi_train:.3f}]$ & "
            f"${fmt_pvalue_tex(w_train.pvalue)}$ \\\\\n"
        )
        fh.write(
            f"Median test-RMSE ratio HBA/\\S\\ref{{sec:tumor_inmune}} & "
            f"${med_test:.3f}$ & "
            f"$[{lo_test:.3f},\\,{hi_test:.3f}]$ & "
            f"${fmt_pvalue_tex(w_test.pvalue)}$ \\\\\n"
        )
        fh.write("\\midrule\n")
        fh.write(
            "$\\tau$ & "
            "Failure rate HBA & Failure rate \\S\\ref{sec:tumor_inmune} & "
            "Failure-rate ratio HBA/\\S\\ref{sec:tumor_inmune} \\\\\n"
        )
        fh.write("\\midrule\n")
        for _, r in sens.iterrows():
            tau_str = f"{r.tau:.2f}"
            is_paper = abs(r.tau - TAU_PAPER) < 1e-9
            tau_cell = f"$\\mathbf{{{tau_str}}}$" if is_paper else f"${tau_str}$"
            p_hba_str = fmt_prop_tex(r.p_hba, r.lo_hba, r.hi_hba)
            p_s7_str = fmt_prop_tex(r.p_s7, r.lo_s7, r.hi_s7)
            ratio_str = fmt_ratio_tex(r.ratio, r.lo_ratio, r.hi_ratio)
            if is_paper:
                p_hba_str = f"\\mathbf{{{p_hba_str}}}"
                p_s7_str = f"\\mathbf{{{p_s7_str}}}"
                ratio_str = f"\\mathbf{{{ratio_str}}}"
            fh.write(
                f"{tau_cell} & "
                f"${p_hba_str}$ & "
                f"${p_s7_str}$ & "
                f"${ratio_str}$ \\\\\n"
            )
        fh.write("\\bottomrule\n")
        fh.write("\\end{tabular}\n")
        fh.write("\\end{table}\n")

    print(TXT_OUT.read_text())


if __name__ == "__main__":
    main()
