"""
preprocess_bruchovsky_offphase.py
==================================

Phase S9.1 (Bruchovsky pivot — OFF-phase variant) — preprocessing of the
Bruchovsky/Tanaka Vancouver Phase II IADT dataset, restricted to the
OFF-treatment block of cycle 1, where PSA REBOUNDS from a low
post-treatment plateau toward the threshold for treatment resumption.

Rationale: the §7 fractional tumor-immune model is structurally designed
for trajectories that ASCEND from a low baseline toward an asymptote
g(y) (in the integral form y = g(y) + I^α(a F(y)), the integral is
non-positive when F < 0 and y(0) < g(y); §7's natural regime is
"growth toward controlled equilibrium"). The ON-phase PSA decay (peak
to plateau) is the OPPOSITE direction and does not fit §7 with the
standard sign conventions. The OFF-phase rebound (plateau to peak)
matches §7's natural growth-toward-asymptote regime.

Pipeline (decisions D1-D7 with D1'' replacing D1' for OFF-phase):

  (D1)   First cycle only:        keep rows with cycle == 1.
  (D1'') OFF-phase only:          filter treatment_on == 0 (NOTE: 0 instead
                                  of 1 with respect to the ON-phase variant).
  (D2)   Drop missing PSA:        drop rows with NaN psa.
  (D3)   Outcomes:                parse classificationsofpatients.txt.
  (D4)   Cohort filter:           >= 8 PSA observations + >= 60 days.
  (Align) sort + t = day - day_min per patient.
  (D5)   Normalisation:           y_norm = psa / max_t(psa) in [0, 1] per
                                  patient. The max in OFF-phase is the
                                  rebound peak at the end of OFF-phase, so
                                  y_norm(t_late) approaches 1 by construction.
  (D6)   Train/test split:        first 70% chronological rank = train.
  (D7)   Nuisance calibration:    per-patient
                                    A_{0,i} = max(y_norm_train) - min(y_norm_train)
                                    c_{0,i} = mean of y_norm_train in last 25%
                                  In OFF-phase, c_{0,i} approximates 1
                                  (the rebound asymptote), which is the §7
                                  asymptote g(y) when tau = 0. This makes the
                                  §7 integral form natural for the data.

Outputs (suffix _offphase to coexist with the ON-phase outputs):
    outputs_clinical/bruchovsky_cohort_filtered_offphase.csv
    outputs_clinical/bruchovsky_per_patient_offphase.csv
    outputs_clinical/bruchovsky_cohort_stats_offphase.txt
    figuras/fig_S9_1_bruchovsky_cohort_offphase.pdf

Usage:
    cd codigo
    python3 preprocess_bruchovsky_offphase.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

MIN_MEASUREMENTS_OFFPHASE = 8
MIN_TIMESPAN_DAYS_OFFPHASE = 60
TRAIN_FRAC = 0.7

REGIME_MAP = {
    "without_relapse":  "eradication",
    "with_relapse":     "dormancy",
    "with_metastasis":  "escape",
}

GLOBAL_NUISANCE = {
    "p":   3.0,
    "mu":  0.1,
    "tau": 0.0,
}

FIG_SIZE = (6.5, 9.0)
REGIME_COLORS = {
    "eradication": "#2E7D32",
    "dormancy":    "#EF6C00",
    "escape":      "#C62828",
}

SUFFIX = "_offphase"


# ---------------------------------------------------------------------------
# 2. Paths
# ---------------------------------------------------------------------------


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def input_csv() -> Path:
    return project_root() / "outputs_dataset" / "bruchovsky_full.csv"


def classifications_txt() -> Path:
    return (project_root() / "outputs_dataset" / "_cache"
            / "dataTanaka_extracted" / "dataTanaka"
            / "classificationsofpatients.txt")


def output_dir() -> Path:
    p = project_root() / "outputs_clinical"
    p.mkdir(parents=True, exist_ok=True)
    return p


def figuras_dir() -> Path:
    p = project_root() / "figuras"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 3. Classification parser
# ---------------------------------------------------------------------------


def parse_classifications(path: Path) -> dict[int, str]:
    if not path.exists():
        sys.exit(f"Classifications file not found at {path}.")
    text = path.read_text(encoding="utf-8", errors="replace")
    head = text.split("Shaw_et_al:")[0]
    out: dict[int, str] = {}
    for category_label, key in [
        ("Without relapse:",                  "without_relapse"),
        ("With metastasis(without relapse):", "with_metastasis"),
        ("With relapse:",                     "with_relapse"),
    ]:
        pat = re.escape(category_label).replace(r"\(", r"\s*\(").replace(r"\)", r"\)")
        m = re.search(pat, head)
        if m is None:
            print(f"  [warn] category {category_label!r} not found")
            continue
        start = m.end()
        rest = head[start:]
        end_match = re.search(r"\n\s*With|\n\s*Without|\n\s*\n\s*\n", rest)
        block = rest[: end_match.start()] if end_match else rest
        ids = [int(x) for x in re.findall(r"\b\d{1,4}\b", block)]
        for pid in ids:
            out[pid] = key
    return out


# ---------------------------------------------------------------------------
# 4. Pipeline
# ---------------------------------------------------------------------------


def load_raw() -> pd.DataFrame:
    csv = input_csv()
    if not csv.exists():
        sys.exit(f"Input CSV not found at {csv}. Run explore_bruchovsky.py first.")
    df = pd.read_csv(csv)
    print(f"  loaded {len(df)} rows, {df['patient_id'].nunique()} patients")
    return df


def step_D1_first_cycle(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["cycle"] == 1].copy()
    print(f"  (D1) first cycle only: kept {len(out)} of {len(df)} rows; "
          f"{out['patient_id'].nunique()} patients")
    return out


def step_D1pp_off_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to rows with treatment_on == 0 (OFF-phase)."""
    out = df[df["treatment_on"] == 0].copy()
    print(f"  (D1'') OFF-phase only: kept {len(out)} of {len(df)} rows; "
          f"{out['patient_id'].nunique()} patients have OFF-phase data in cycle 1")
    return out


def step_D2_drop_missing_psa(df: pd.DataFrame) -> pd.DataFrame:
    out = df.dropna(subset=["psa"]).copy()
    print(f"  (D2) drop missing PSA: kept {len(out)} of {len(df)} rows")
    return out


def step_D3_outcomes(df: pd.DataFrame, classif: dict[int, str]) -> pd.DataFrame:
    df = df.copy()
    df["clinical_outcome"] = df["patient_id"].map(classif)
    df["regime"] = df["clinical_outcome"].map(REGIME_MAP)
    n_with = df["clinical_outcome"].notna().sum()
    print(f"  (D3) outcomes annotated: {n_with}/{len(df)} rows have category")
    no_class = df.loc[df["clinical_outcome"].isna(), "patient_id"].unique()
    if len(no_class) > 0:
        print(f"    [warn] {len(no_class)} patients missing classification")
    return df


def step_D4_cohort_filter(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("patient_id").size()
    spans = df.groupby("patient_id")["day"].agg(lambda s: int(s.max() - s.min()))
    keep = (counts >= MIN_MEASUREMENTS_OFFPHASE) & (spans >= MIN_TIMESPAN_DAYS_OFFPHASE)
    keep_ids = counts.index[keep].tolist()
    out = df[df["patient_id"].isin(keep_ids)].copy()
    print(f"  (D4) cohort filter: kept {len(keep_ids)} of {df['patient_id'].nunique()} "
          f"patients (>= {MIN_MEASUREMENTS_OFFPHASE} obs AND "
          f">= {MIN_TIMESPAN_DAYS_OFFPHASE} days OFF)")
    return out


def step_align_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["patient_id", "day"]).reset_index(drop=True)
    t0 = df.groupby("patient_id")["day"].transform("min")
    df["t"] = (df["day"] - t0).astype(float)
    print(f"  time alignment: t range [{df['t'].min():.0f}, {df['t'].max():.0f}] days")
    return df


def step_D5_normalize(df: pd.DataFrame) -> pd.DataFrame:
    psa_max = df.groupby("patient_id")["psa"].transform("max")
    df["y_norm"] = (df["psa"] / psa_max).astype(float)
    print(f"  (D5) normalisation: y_norm in "
          f"[{df['y_norm'].min():.4f}, {df['y_norm'].max():.4f}]")
    return df


def step_D6_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["patient_id", "t"]).reset_index(drop=True)
    splits = []
    for _, sub in df.groupby("patient_id", sort=False):
        n = len(sub)
        n_train = max(1, int(np.ceil(TRAIN_FRAC * n)))
        n_train = min(n_train, n - 1)
        splits.extend(["train"] * n_train + ["test"] * (n - n_train))
    df["split"] = splits
    print(f"  (D6) train/test split: {(df['split']=='train').sum()} train, "
          f"{(df['split']=='test').sum()} test")
    return df


def step_D7_nuisance(df: pd.DataFrame) -> dict[int, dict]:
    out = {}
    for pid, sub in df.groupby("patient_id", sort=False):
        train = sub[sub["split"] == "train"].sort_values("t")
        if len(train) == 0:
            out[pid] = {"A0_calib": np.nan, "c0_calib": np.nan}
            continue
        A0 = float(train["y_norm"].max() - train["y_norm"].min())
        n_late = max(1, int(np.ceil(0.25 * len(train))))
        c0 = float(train["y_norm"].iloc[-n_late:].mean())
        out[pid] = {"A0_calib": A0, "c0_calib": c0}
    a0s = [v["A0_calib"] for v in out.values() if not np.isnan(v["A0_calib"])]
    c0s = [v["c0_calib"] for v in out.values() if not np.isnan(v["c0_calib"])]
    print(f"  (D7) nuisance: A_0 in [{min(a0s):.3f}, {max(a0s):.3f}], "
          f"c_0 in [{min(c0s):.3f}, {max(c0s):.3f}]")
    return out


# ---------------------------------------------------------------------------
# 5. Per-patient summary
# ---------------------------------------------------------------------------


def collect_per_patient(df: pd.DataFrame, nuisance: dict[int, dict]
                        ) -> pd.DataFrame:
    rows = []
    for pid, sub in df.groupby("patient_id", sort=False):
        sub = sub.sort_values("t")
        psa_first = float(sub["psa"].iloc[0])
        psa_last = float(sub["psa"].iloc[-1])
        rows.append({
            "patient_id": pid,
            "n_measurements": len(sub),
            "t_max": float(sub["t"].max()),
            "psa_first": psa_first,
            "psa_last": psa_last,
            "ratio_last_first": (psa_last / psa_first) if psa_first > 0 else float("nan"),
            "clinical_outcome": sub["clinical_outcome"].iloc[0],
            "regime": sub["regime"].iloc[0],
            "A0_calib": nuisance[pid]["A0_calib"],
            "c0_calib": nuisance[pid]["c0_calib"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Outputs
# ---------------------------------------------------------------------------


def write_long_csv(df: pd.DataFrame) -> Path:
    out = df[["patient_id", "t", "psa", "y_norm", "split",
              "clinical_outcome", "regime"]].copy()
    p = output_dir() / f"bruchovsky_cohort_filtered{SUFFIX}.csv"
    out.to_csv(p, index=False)
    print(f"  wrote {p} ({len(out)} rows)")
    return p


def write_per_patient_csv(per_pat: pd.DataFrame) -> Path:
    p = output_dir() / f"bruchovsky_per_patient{SUFFIX}.csv"
    per_pat.to_csv(p, index=False)
    print(f"  wrote {p} ({len(per_pat)} patients)")
    return p


def write_stats(df: pd.DataFrame, per_pat: pd.DataFrame) -> Path:
    p = output_dir() / f"bruchovsky_cohort_stats{SUFFIX}.txt"
    with open(p, "w") as f:
        f.write("Phase S9.1 (Bruchovsky pivot — OFF-phase) cohort summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Pipeline: D1 cycle==1, D1'' treatment_on==0,\n"
                f"D2 drop NaN PSA, D3 outcomes from classifications,\n"
                f"D4 filter >= {MIN_MEASUREMENTS_OFFPHASE} obs AND "
                f">= {MIN_TIMESPAN_DAYS_OFFPHASE} days,\n"
                f"D5 norm by max(PSA), D6 split "
                f"{TRAIN_FRAC*100:.0f}%/{(1-TRAIN_FRAC)*100:.0f}%,\n"
                f"D7 per-patient nuisance + global nuisance.\n\n")

        f.write(f"N patients (final cohort): {df['patient_id'].nunique()}\n")
        f.write(f"Total measurements (cycle 1 OFF, PSA non-missing): {len(df)}\n")
        f.write(f"Train / test rows: {(df['split']=='train').sum()} / "
                f"{(df['split']=='test').sum()}\n\n")

        f.write("Distribution of OFF-phase measurements per patient:\n")
        f.write(df.groupby("patient_id").size().describe().to_string())
        f.write("\n\n")

        f.write("Distribution of OFF-phase t_max (days) per patient:\n")
        f.write(per_pat["t_max"].describe().to_string())
        f.write("\n\n")

        f.write("Distribution of PSA (ng/mL) in OFF-phase:\n")
        f.write(df["psa"].describe().to_string())
        f.write("\n\n")

        f.write("Distribution of regime:\n")
        for r in ["eradication", "dormancy", "escape"]:
            n_r = (per_pat["regime"] == r).sum()
            pct = 100 * n_r / len(per_pat)
            f.write(f"  {r:12s}: {n_r:3d} ({pct:5.1f}%)\n")
        f.write("\n")

        f.write("Per-patient nuisance calibration:\n")
        a0 = per_pat["A0_calib"].dropna()
        c0 = per_pat["c0_calib"].dropna()
        f.write(f"  A_0:  mean={a0.mean():.3f}, std={a0.std():.3f}, "
                f"range=[{a0.min():.3f}, {a0.max():.3f}]\n")
        f.write(f"  c_0:  mean={c0.mean():.3f}, std={c0.std():.3f}, "
                f"range=[{c0.min():.3f}, {c0.max():.3f}]\n\n")

        f.write("Global nuisance:\n")
        for k, v in GLOBAL_NUISANCE.items():
            f.write(f"  {k:6s} = {v}\n")

        f.write("\nCross-tab clinical_outcome vs regime:\n")
        f.write(pd.crosstab(per_pat["clinical_outcome"],
                            per_pat["regime"]).to_string())
        f.write("\n")
    print(f"  wrote {p}")
    return p


def make_figure(df: pd.DataFrame, per_pat: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=FIG_SIZE, sharex=False)
    regimes = ["eradication", "dormancy", "escape"]
    for ax, regime in zip(axes, regimes):
        color = REGIME_COLORS[regime]
        regime_pids = per_pat.loc[per_pat["regime"] == regime, "patient_id"]
        n_in = 0
        for pid in regime_pids:
            sub = df[df["patient_id"] == pid].sort_values("t")
            if len(sub) == 0:
                continue
            n_in += 1
            train = sub[sub["split"] == "train"]
            test  = sub[sub["split"] == "test"]
            ax.plot(train["t"], train["y_norm"], "-", color=color,
                    alpha=0.40, lw=0.8)
            if len(test) > 0:
                bridge_t = list(train["t"].iloc[-1:]) + list(test["t"])
                bridge_y = list(train["y_norm"].iloc[-1:]) + list(test["y_norm"])
                ax.plot(bridge_t, bridge_y, "--", color=color,
                        alpha=0.40, lw=0.8)
            ax.plot(sub["t"], sub["y_norm"], "o",
                    color=color, alpha=0.50, ms=2.4)
        ax.set_title(f"{regime.capitalize()} (n = {n_in})",
                     fontsize=11, color=color)
        ax.set_ylabel(r"$y_{\mathrm{norm}}(t) = \mathrm{PSA}(t)/\max\,\mathrm{PSA}$",
                      fontsize=10)
        ax.grid(True, alpha=0.25, lw=0.4)
        ax.set_ylim(-0.02, 1.05)
    axes[-1].set_xlabel(r"$t$ (days from start of cycle 1 OFF-phase)", fontsize=10)
    fig.suptitle(r"Bruchovsky Phase II IADT cohort, cycle 1 OFF-phase rebound trajectories grouped by clinical regime",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = figuras_dir() / f"fig_S9_1_bruchovsky_cohort{SUFFIX}.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")
    return p


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Phase S9.1 (Bruchovsky pivot — OFF-PHASE)")
    print("=" * 60)

    print("\n[load] reading raw CSV...")
    df = load_raw()

    print("\n[parse] reading classifications file...")
    classif = parse_classifications(classifications_txt())
    print(f"  parsed {len(classif)} classifications")

    print("\n[step D1] filtering to first cycle...")
    df = step_D1_first_cycle(df)

    print("\n[step D1''] filtering to OFF-phase only (treatment_on == 0)...")
    df = step_D1pp_off_phase(df)

    print("\n[step D2] dropping missing PSA...")
    df = step_D2_drop_missing_psa(df)

    print("\n[step D3] annotating outcomes...")
    df = step_D3_outcomes(df, classif)

    print("\n[step D4] cohort filter...")
    df = step_D4_cohort_filter(df)

    print("\n[align] aligning time...")
    df = step_align_time(df)

    print("\n[step D5] normalising PSA...")
    df = step_D5_normalize(df)

    print("\n[step D6] train/test split...")
    df = step_D6_split(df)

    print("\n[step D7] nuisance calibration...")
    nuisance = step_D7_nuisance(df)

    print("\n[collect] per-patient summary...")
    per_pat = collect_per_patient(df, nuisance)

    print("\n[output] writing CSVs and stats...")
    write_long_csv(df)
    write_per_patient_csv(per_pat)
    write_stats(df, per_pat)

    print("\n[figure] rendering 3-panel cohort trajectories (OFF-phase)...")
    make_figure(df, per_pat)

    print("\n" + "=" * 60)
    print(f"Done. Final cohort: {df['patient_id'].nunique()} patients, "
          f"{len(df)} OFF-phase PSA measurements.")
    print("Inspect outputs_clinical/bruchovsky_cohort_stats_offphase.txt and "
          "figuras/fig_S9_1_bruchovsky_cohort_offphase.pdf.")
    print("=" * 60)


if __name__ == "__main__":
    main()
