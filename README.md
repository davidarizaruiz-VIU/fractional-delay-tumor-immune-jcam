# A fractional-delay asymptotic framework with physics-informed learning for biomedical tumor–immune dynamics

Code accompanying the paper:

> Ariza-Ruiz, D. (2026). *A fractional-delay asymptotic framework with
> physics-informed learning for biomedical tumor–immune dynamics.*
> Journal of Computational and Applied Mathematics, Special Issue
> *VSI:CAM_Math AI & Biomed Apps*. (Submitted)

This repository implements the analytical, computational and
inferential pipeline of the paper: the L1 discretisation of the
right-sided Caputo derivative, the fractional physics-informed neural
network (fPINN) with asymptotically-aware ansatz, the patient-level
Picard profile-likelihood calibration on the Bruchovsky intermittent
androgen-deprivation therapy (IADT) cohort, the *a priori*
amplitude-bound verification, the head-to-head benchmark against the
Hirata–Bruchovsky–Aihara (2010) compartmental ODE, and the paired
bootstrap/Wilcoxon inference layer that produces the confidence
intervals reported in §9.

## Repository structure

```
Github/
├── README.md                         this file
├── LICENSE                           MIT
├── requirements.txt                  Python dependencies
├── .gitignore                        standard Python + LaTeX
├── data/
│   └── README.md                     instructions for obtaining the
│                                     Bruchovsky IADT cohort
├── src/
│   ├── preprocess/
│   │   └── preprocess_bruchovsky_offphase.py
│   │           cycle-1 OFF-phase extraction, cohort filter,
│   │           amplitude-ratio normalisation, train/test split
│   ├── calibration/
│   │   ├── calibrate_bruchovsky.py
│   │   │       per-patient profile-likelihood calibration of the
│   │   │       fractional integral equation under the linear-affine
│   │   │       tumor–immune restriction (damped Picard iteration,
│   │   │       T_max = 320 d)
│   │   └── calibrate_bruchovsky_Tmax_invariance.py
│   │           re-calibration with extended mesh T_max = 2000 d for
│   │           the T_max-invariance check
│   ├── benchmark/
│   │   ├── benchmark_hba_exact.py
│   │   │       head-to-head benchmark against the exact
│   │   │       Hirata–Bruchovsky–Aihara (2010) three-compartment
│   │   │       piecewise-linear ODE
│   │   └── benchmark_hba_paired_inference.py
│   │           paired-inference layer over benchmark_hba_exact.py:
│   │           bootstrap 95% CI for the median per-patient RMSE
│   │           ratio, Wilcoxon signed-rank tests, and sensitivity
│   │           of the failure-rate ratio to the threshold τ
│   ├── numerical_schemes/
│   │   ├── classical_scheme.py
│   │   │       L1 discretisation of the right-sided Caputo
│   │   │       derivative on a truncated mesh [0, T_max]
│   │   ├── fpinn.py
│   │   │       fractional physics-informed neural network for the
│   │   │       direct and inverse problems
│   │   ├── run_l1_convergence_study.py
│   │   │       convergence-order study of the L1 scheme on a
│   │   │       manufactured solution
│   │   └── run_fpinn_forward_study.py
│   │           fPINN forward-problem verification on the
│   │           manufactured solution and on the §7 instance
│   └── analysis/
│       ├── verify_amplitude_bound_sharpness.py
│       │       direct linear-solve computation of the cohort-sharpened
│       │       saturation factor η for both the original and the
│       │       time-scaled kernels
│       └── make_cohort_amplitude_figure.py
│               cohort-wide overview figure: data amplitude vs the
│               framework's representability bounds
└── tests/
    └── test_classical_scheme.py
            unit tests for the L1 scheme on the manufactured solution
```

## Reproducing the paper

Each script is self-contained and can be invoked directly. The
typical end-to-end pipeline is:

```bash
# 1. Preprocess the Bruchovsky cohort (requires R package tumgr;
#    see data/README.md for instructions).
python -m src.preprocess.preprocess_bruchovsky_offphase

# 2. Run the patient-level calibration of the fractional framework.
python -m src.calibration.calibrate_bruchovsky

# 3. Verify T_max-invariance of the calibration (optional, slower).
python -m src.calibration.calibrate_bruchovsky_Tmax_invariance

# 4. Run the head-to-head benchmark against HBA-2010.
python -m src.benchmark.benchmark_hba_exact

# 5. Compute the paired-inference layer (bootstrap CIs +
#    Wilcoxon signed-rank + threshold sensitivity) over (4).
python -m src.benchmark.benchmark_hba_paired_inference

# 6. Compute the η saturation factors for both kernels.
python -m src.analysis.verify_amplitude_bound_sharpness

# 7. Generate the cohort-amplitude overview figure.
python -m src.analysis.make_cohort_amplitude_figure
```

The convergence-order and fPINN forward studies (used for the
verification in §7 of the paper) can be run independently:

```bash
python -m src.numerical_schemes.run_l1_convergence_study
python -m src.numerical_schemes.run_fpinn_forward_study
```

Outputs (CSV tables, statistics text files, vector PDF figures) are
written to a local `./outputs/` directory by each script.

## Hardware and runtime

The complete pipeline (steps 1–6 above) runs in approximately
2–3 hours on a modern laptop CPU (Apple M-series, 16 GB RAM); the
T_max-invariance step (3) accounts for the bulk of the wall time
(75–100 min) and is optional. No GPU is required; the fPINN training
in `fpinn.py` uses CPU-only PyTorch by default.

## Dependencies

See `requirements.txt`. Core stack:

- Python ≥ 3.10
- NumPy, SciPy, pandas, matplotlib
- PyTorch (CPU build is sufficient)

The patient-level cohort data require the R package
[`tumgr`](https://cran.r-project.org/package=tumgr) (see
`data/README.md`).

## License

MIT — see `LICENSE`.

## Citation

```bibtex
@article{ArizaRuiz2026FractionalTumorImmune,
  author  = {Ariza-Ruiz, D.},
  title   = {A fractional-delay asymptotic framework with
             physics-informed learning for biomedical
             tumor--immune dynamics},
  journal = {Journal of Computational and Applied Mathematics},
  year    = {2026},
  note    = {Special Issue VSI:CAM\_Math AI \& Biomed Apps,
             submitted}
}
```
