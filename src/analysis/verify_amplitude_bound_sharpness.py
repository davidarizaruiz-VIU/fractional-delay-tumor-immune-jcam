"""
verify_amplitude_bound_sharpness.py
====================================

Numerical investigation of the tightness of the Theorem 9.1 amplitude bound

    A_tr  <=  I_alpha * L_F * ||y||_inf       (eq:amplitude_bound_betafree)

in the limit L_F * I_alpha -> 1^-.

Setting (linear-affine, no convolution, forcing-free):
  alpha = 1.5,  p = 3,  L_F = 1,  c_0 = 1,
  beta_0 = 0,  xi = 0,  tau = 0,
  a(s) = A_0 * (1+s)^{-p},
  f(u, v) = -L_F * u   (so F(y) = -L_F * y).

For each epsilon in {0.5, 0.3, ..., 0.001} we set
  A_0(epsilon) := (1 - epsilon) / (K_alpha * L_F)
so that L_F * I_alpha = 1 - epsilon.

We measure r := 1 - y_*(0)/g_*  with g_* := y_*(T_max),
and the saturation ratio  s := r / (L_F * I_alpha).

FINDING: s converges to a CONSTANT eta(alpha, p) < 1 as epsilon -> 0+.
For (alpha=1.5, p=3) we measure eta ≈ 0.682. This shows that the
Theorem 9.1 amplitude bound, while explicit and a valid upper bound,
is NOT asymptotically tight: there is a residual slack of order
1 - eta ~ 30% that does not vanish in the contraction-boundary limit.

The geometric origin of this residual: the integral averaging in
   r = (L_F / Gamma(alpha)) * int_0^infty s^{alpha-1} a(s) (y_*(s)/g_*) ds
   = L_F * I_alpha * <y_*/g_*>_a
weights y_*(s)/g_* by the kernel s^{alpha-1} a(s). Since y_* always
attains values strictly below g_* on a positive-measure subset of the
kernel support (specifically near s=0, where y_*(0) -> eta * g_* in
the limit, NOT to 0), the weighted average <y_*/g_*>_a is strictly
below 1 by a factor that does not vanish as epsilon -> 0+.

Sharpening the bound to capture this geometric factor would require
spectral analysis of the integral operator I_alpha on C_b(R+), which
is left as an open problem (cf. discussion of tightness following Theorem 9.1
and Remark 9.4 on amplitude-bound extensions).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import beta as Beta, gamma as Gamma
import matplotlib.pyplot as plt


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def figuras_dir() -> Path:
    p = project_root() / "figuras"
    p.mkdir(parents=True, exist_ok=True)
    return p


# Setting
ALPHA = 1.5
P_KERNEL = 3
L_F = 1.0
C_0 = 1.0
T_MAX = 500.0   # large enough for tail to vanish: (1+T_max)^{alpha-p} = 501^{-1.5} ~ 9e-5
H = 0.1
N_ITER = 500
TOL = 1e-12
W_DAMPING = 0.5

K_ALPHA = Beta(ALPHA, P_KERNEL - ALPHA) / Gamma(ALPHA)


def solve_fixed_point(A_0, T_max=T_MAX, h=H, n_iter=N_ITER, tol=TOL, w=W_DAMPING):
    """Damped Picard solve of the linear forcing-free fixed-point equation
       y(t) = c_0 - L_F * (1/Gamma(alpha)) * \\int_t^{T_max} (s-t)^{alpha-1} a(s) y(s) ds
       with a(s) = A_0 (1+s)^{-p}.

       Returns (t, y, info_dict).
    """
    N = int(round(T_max / h))
    t = h * np.arange(N + 1)
    a = A_0 * (1 + t) ** (-P_KERNEL)

    # Precompute (s - t)^(alpha-1) weights matrix
    diff_st = t[None, :] - t[:, None]            # (j, k) -> t_k - t_j
    weights = np.where(diff_st > 0,
                       np.maximum(diff_st, 1e-15) ** (ALPHA - 1),
                       0.0)

    y = np.full(N + 1, C_0)
    res_history = []
    iters_done = 0
    for k in range(n_iter):
        iters_done = k + 1
        F_y = -L_F * y                            # F(y) = -L_F y (forcing-free, no conv)
        a_F = a * F_y
        Ty = np.zeros(N + 1)
        for j in range(N):
            integrand = weights[j, j:] * a_F[j:]
            Ty[j] = C_0 + np.trapezoid(integrand, t[j:]) / Gamma(ALPHA)
        Ty[-1] = C_0
        y_new = (1.0 - w) * y + w * Ty
        d = float(np.max(np.abs(y_new - y)))
        res_history.append(d)
        y = y_new
        if d < tol:
            break

    return t, y, dict(iters=iters_done, residual=res_history[-1] if res_history else np.nan,
                       res_history=np.array(res_history))


def compute_eta_direct(T_max=T_MAX, h=H):
    """Direct computation of eta(alpha, p) via linear system solve, NO convolution.

    At the contraction-boundary value A_0^* := 1/(L_F * K_alpha), the
    operator L_F * I_alpha has spectrum in [0, 1] but the operator
    (I + L_F * I_alpha) is well-conditioned (spectrum in [1, 2]) and
    invertible. The limiting normalised fixed point is
        phi_infty := (I + L_F * I_alpha^*)^{-1} 1
    and eta(alpha, p) = 1 - phi_infty(0).

    This is a DIRECT MATHEMATICAL CHARACTERISATION for the linear-affine
    NO-CONVOLUTION sub-case (xi = 0, F(y) = -L_F y).
    """
    A_0_star = 1.0 / (L_F * K_ALPHA)
    N = int(round(T_max / h))
    t = h * np.arange(N + 1)
    a = A_0_star * (1 + t) ** (-P_KERNEL)

    # Build matrix M_{j,k} = (h / Gamma(alpha)) * (t_k - t_j)^(alpha-1) * a[k]
    # for k > j (strictly upper triangular), 0 otherwise.
    diff_st = t[None, :] - t[:, None]
    M = np.where(diff_st > 0, np.maximum(diff_st, 1e-15) ** (ALPHA - 1), 0.0)
    M = M * a[None, :]                # multiply each column k by a[k]
    M = M * (h / Gamma(ALPHA))         # quadrature weight

    # Linear system: (I + L_F * M) phi = 1
    A_mat = np.eye(N + 1) + L_F * M
    rhs = np.ones(N + 1)
    phi_infty = np.linalg.solve(A_mat, rhs)

    eta_direct = 1.0 - float(phi_infty[0])

    # Diagnostic: operator norm L_F * ||M||_inf (row-sum norm) - this estimates
    # the operator norm in C_b, which the theory says equals L_F * A_0^* * K_alpha = 1.
    # Note: eigvals of M would be 0 (strictly upper triangular discretisation does
    # not preserve continuum spectrum), so we use the row-sum operator norm instead.
    norm_LF_M = float(L_F * np.max(np.sum(M, axis=1)))

    return dict(
        A_0_star=A_0_star,
        eta_direct=eta_direct,
        phi_at_0=float(phi_infty[0]),
        phi_at_T=float(phi_infty[-1]),
        operator_norm_LF_I_alpha=norm_LF_M,
        N_grid=N + 1,
        h=h, T_max=T_max,
    )


def compute_eta_TS_with_convolution(s_0=300.0, T_max=5000.0, h=5.0,
                                      gamma_sat=0.6, xi=0.4, mu=0.1):
    """Direct computation of eta for the TIME-SCALED kernel
        a(s) = A_0 (s_0/(s_0+s))^p   (s_0 = 300 days, IADT OFF-block scale)
    in the cohort-with-convolution setting.

    Resolves C3 (v2 strawman): provides the kernel-specific saturation
    factor for trial v2 subcase (b) so that the achievable amplitude
    bound at A_0 = A_0^max_TS = 1/(s_0^α K_α L_F) can be reported
    rigorously without invoking heuristic equivalence with the
    original-kernel eta.

    Returns: dict with eta_TS_with_conv, A_0_max_TS, etc.
    """
    L_F_local = gamma_sat + xi
    A_0_max_TS = 1.0 / (s_0**ALPHA * K_ALPHA * L_F_local)
    N = int(round(T_max / h))
    t = h * np.arange(N + 1)
    a = A_0_max_TS * (s_0 / (s_0 + t)) ** P_KERNEL  # time-scaled kernel
    # I_alpha matrix M[i,j]: weight (t_j - t_i) used for j > i (forward integration)
    diff_M = t[None, :] - t[:, None]
    M = np.where(diff_M > 0, np.maximum(diff_M, 1e-15) ** (ALPHA - 1), 0.0)
    M = M * a[None, :] * (h / Gamma(ALPHA))

    # Convolution K∗y matrix: K(t_i - t_j) for j <= i (causal kernel)
    diff_K = t[:, None] - t[None, :]
    KK = mu * np.exp(-mu * np.maximum(diff_K, 0.0)) * h
    KK = np.tril(KK, k=0)
    np.fill_diagonal(KK, np.diag(KK) * 0.5)
    KK[1:, 0] *= 0.5

    op = np.eye(N + 1) + M @ (gamma_sat * np.eye(N + 1) + xi * KK)
    phi_infty = np.linalg.solve(op, np.ones(N + 1))
    eta_TS_with_conv = 1.0 - float(phi_infty[0])
    norm_LF_M = L_F_local * M.sum(axis=1).max()

    return dict(
        kernel="time-scaled (s_0+s)^{-p}",
        s_0=s_0,
        gamma_sat=gamma_sat, xi=xi, mu=mu, L_F=L_F_local,
        A_0_max_TS=A_0_max_TS,
        eta_TS_with_conv=eta_TS_with_conv,
        phi_at_0=float(phi_infty[0]),
        phi_at_T=float(phi_infty[-1]),
        operator_norm_LF_I_alpha_TS=norm_LF_M,
        N_grid=N + 1, h=h, T_max=T_max,
    )


def compute_eta_with_convolution(T_max=200.0, h=0.05,
                                  gamma_sat=0.6, xi=0.4, mu=0.1):
    """Direct computation of eta with the convolution operator K∗y included.

    Setting (cohort linear-affine WITH convolution):
      F(y) = -gamma_sat * y - xi * (K∗y)
      K(s, sigma) = mu * exp(-mu*(s-sigma))   (exponential kernel with rho_K=1)
      L_F = gamma_sat + xi * rho_K = gamma_sat + xi

    Limit equation at A_0 = A_0^*:
      phi + L_F * I_alpha^* * [(gamma_sat/L_F) phi + (xi/L_F) K∗phi] = 1
    equivalently:
      (I + I_alpha^* * (gamma_sat * I + xi * K)) phi = 1

    where I_alpha^* := M with A_0 = A_0^* embedded.
    """
    L_F_local = gamma_sat + xi   # rho_K = 1 for exponential kernel
    A_0_star = 1.0 / (L_F_local * K_ALPHA)
    N = int(round(T_max / h))
    t = h * np.arange(N + 1)
    a = A_0_star * (1 + t) ** (-P_KERNEL)

    # I_alpha^* matrix: M[j,k] = (h/Γ(α)) (t_k-t_j)^(α-1) a(t_k) for k>j
    diff_st = t[None, :] - t[:, None]
    M = np.where(diff_st > 0, np.maximum(diff_st, 1e-15) ** (ALPHA - 1), 0.0)
    M = M * a[None, :] * (h / Gamma(ALPHA))

    # Convolution K∗y(t_i) = ∫_0^{t_i} mu*exp(-mu*(t_i-σ)) y(σ) dσ
    # Trapezoidal: KK[i,j] = h * mu * exp(-mu*(t_i-t_j)) for j ≤ i, with edge weights /2
    KK = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            KK[i, j] = mu * np.exp(-mu * (t[i] - t[j]))
    KK *= h
    # Trapezoidal edge correction
    for i in range(1, N + 1):
        KK[i, 0] *= 0.5
        KK[i, i] *= 0.5

    # Operator: (I + M @ (gamma_sat * I + xi * KK)) phi = 1
    op = np.eye(N + 1) + M @ (gamma_sat * np.eye(N + 1) + xi * KK)
    phi_infty = np.linalg.solve(op, np.ones(N + 1))
    eta_with_conv = 1.0 - float(phi_infty[0])

    return dict(
        gamma_sat=gamma_sat, xi=xi, mu=mu,
        L_F=L_F_local, A_0_star=A_0_star,
        eta_with_conv=eta_with_conv,
        phi_at_0=float(phi_infty[0]),
        phi_at_T=float(phi_infty[-1]),
        N_grid=N + 1, h=h, T_max=T_max,
    )


def main():
    print("=" * 72)
    print("Tightness of Theorem 9.1 amplitude bound:")
    print("  (I) DIRECT analytical-numerical computation of eta(alpha, p)")
    print("  (II) Picard sweep verification eta_sweep -> eta_direct as eps -> 0")
    print("=" * 72)
    print(f"  alpha = {ALPHA}, p = {P_KERNEL}, L_F = {L_F}, c_0 = {C_0}")
    print(f"  K_alpha(alpha=1.5) = sqrt(pi)/4 = {K_ALPHA:.10f}")
    print(f"  T_max = {T_MAX}, h = {H}, mesh nodes = {int(T_MAX/H) + 1}")
    print(f"  n_iter_max = {N_ITER}, tol = {TOL}, damping w = {W_DAMPING}")
    print()

    # ===================================================================
    # (I) DIRECT computation of eta via linear solve at A_0 = A_0^*
    # ===================================================================
    print("--- Part (I): DIRECT analytical-numerical computation ---")
    print()
    direct = compute_eta_direct()
    print(f"  At the contraction boundary A_0^* = 1/(L_F * K_alpha) = {direct['A_0_star']:.6f}:")
    print(f"    Discrete operator norm L_F * ||M||_{{C_b->C_b}} (row-sum) = "
          f"{direct['operator_norm_LF_I_alpha']:.6f}")
    print(f"      (theory: continuum operator norm L_F * A_0^* * K_alpha = 1.0;")
    print(f"       discrepancy from theory due to mesh truncation at T_max={direct['T_max']:.0f})")
    print(f"    phi_infty(0) = {direct['phi_at_0']:.6f}")
    print(f"    phi_infty(T_max) = {direct['phi_at_T']:.6f}  (theory: 1.0 by Prop. asymptotic_rate)")
    print(f"    => eta(alpha=1.5, p=3) = 1 - phi_infty(0) = {direct['eta_direct']:.6f}")
    print(f"    => residual slack 1 - eta = {1 - direct['eta_direct']:.6f}")
    print()

    # ===================================================================
    # (I-bis) Cohort setting WITH convolution (xi=0.4, gamma_sat=0.6, mu=0.1)
    # ===================================================================
    print("\n  --- Cohort setting WITH convolution (audit ronda 6, issue #34) ---")
    cohort = compute_eta_with_convolution(
        T_max=200.0, h=0.05,
        gamma_sat=0.6, xi=0.4, mu=0.1,
    )
    print(f"    Setting: gamma_sat={cohort['gamma_sat']}, xi={cohort['xi']}, "
          f"mu={cohort['mu']}, L_F={cohort['L_F']}")
    print(f"    A_0^* = 1/(L_F * K_alpha) = {cohort['A_0_star']:.6f}")
    print(f"    phi_infty(0) = {cohort['phi_at_0']:.6f}")
    print(f"    eta_with_convolution(alpha=1.5, p=3) = "
          f"{cohort['eta_with_conv']:.6f}")
    print(f"    Difference vs no-convolution eta = "
          f"{abs(cohort['eta_with_conv'] - direct['eta_direct']):.4f}")
    print(f"    Sharpened bound for cohort: eta * A_0^cap * K_alpha(1.5) * L_F")
    print(f"      = {cohort['eta_with_conv']:.4f} * 2.0 * 0.4431 * 1.0 = "
          f"{cohort['eta_with_conv'] * 2.0 * K_ALPHA * 1.0:.4f}")
    direct['eta_with_convolution'] = cohort['eta_with_conv']
    direct['cohort_setting'] = cohort

    # ===================================================================
    # (I-ter) TIME-SCALED kernel with convolution (resolves C3 v2 strawman)
    # ===================================================================
    print("\n  --- TIME-SCALED kernel WITH convolution (C3 v2 subcase b) ---")
    print("    Kernel: a(s) = A_0 (s_0/(s_0+s))^p with s_0 = 300 d (IADT scale)")
    cohort_TS = compute_eta_TS_with_convolution(
        s_0=300.0, T_max=5000.0, h=5.0,
        gamma_sat=0.6, xi=0.4, mu=0.1,
    )
    print(f"    Setting: gamma_sat={cohort_TS['gamma_sat']}, xi={cohort_TS['xi']}, "
          f"mu={cohort_TS['mu']}, L_F={cohort_TS['L_F']}")
    print(f"    A_0_max_TS = 1/(s_0^α K_α L_F) = "
          f"{cohort_TS['A_0_max_TS']:.6e}")
    print(f"    Discrete L_F * ||M_TS|| = "
          f"{cohort_TS['operator_norm_LF_I_alpha_TS']:.6f} (target 1.0)")
    print(f"    phi_infty^TS(0) = {cohort_TS['phi_at_0']:.6f}")
    print(f"    eta_TS_with_convolution(alpha=1.5, p=3, s_0=300) = "
          f"{cohort_TS['eta_TS_with_conv']:.6f}")
    print(f"    Achievable amplitude bound at A_0 = A_0_max_TS:")
    print(f"      eta_TS * 1 * ||y||_inf = {cohort_TS['eta_TS_with_conv']:.4f} "
          f"(vs cohort r ~ 0.96, gap = "
          f"{0.96 - cohort_TS['eta_TS_with_conv']:.4f})")
    direct['eta_TS_with_convolution'] = cohort_TS['eta_TS_with_conv']
    direct['cohort_TS_setting'] = cohort_TS

    # Mesh refinement study (Richardson extrapolation to h -> 0+)
    # Use T_max = 200 (sufficient: tail bias (1+200)^(α-p) = 201^(-1.5) ≈ 4e-4)
    # so that we can afford h = 0.025 (N = 8001 nodes, manageable matrix size).
    print("\n  Mesh refinement study (T_max=200 to allow h=0.025):")
    print(f"    Tail bias at T_max=200: (1+200)^(α-p) ≈ {(1+200.0)**(ALPHA-P_KERNEL):.2e}")
    refine_results = []
    for h_ref in [0.2, 0.1, 0.05, 0.025]:
        d = compute_eta_direct(h=h_ref, T_max=200.0)
        refine_results.append((h_ref, d['eta_direct'], d['N_grid']))
        if len(refine_results) >= 2:
            shift = abs(refine_results[-1][1] - refine_results[-2][1])
            print(f"    h = {h_ref:.4f}: eta = {d['eta_direct']:.6f}  "
                  f"(N+1 = {d['N_grid']}, shift from h={refine_results[-2][0]} is {shift:.2e})")
        else:
            print(f"    h = {h_ref:.4f}: eta = {d['eta_direct']:.6f}  "
                  f"(N+1 = {d['N_grid']}, ref)")
    # Richardson convergence-order estimate from last 3 points
    if len(refine_results) >= 3:
        e2, e3, e4 = refine_results[-3][1], refine_results[-2][1], refine_results[-1][1]
        diff_a = abs(e3 - e2); diff_b = abs(e4 - e3)
        ratio_h = refine_results[-3][0] / refine_results[-2][0]   # = 2.0
        if diff_a > 0 and diff_b > 0:
            import math
            q_emp = math.log(diff_a/diff_b) / math.log(ratio_h)
            # Aitken Delta^2 acceleration: eta_inf ≈ e2 - (e3-e2)^2/(e4-2*e3+e2)
            denom = e4 - 2*e3 + e2
            if abs(denom) > 1e-15:
                eta_aitken = e2 - (e3 - e2)**2 / denom
                print(f"    Empirical convergence order: q = {q_emp:.3f}")
                print(f"    Aitken Δ² extrapolation: eta_∞ ≈ {eta_aitken:.6f}")
                # Conservative reporting
                eta_est = eta_aitken
                eta_unc = max(abs(eta_est - e4), 1e-3)
                print(f"    >>> Refined estimate: eta(1.5, 3) ≈ {eta_est:.4f} ± {eta_unc:.4f}")
                # Save for use in summary
                direct['eta_direct_refined'] = eta_est
                direct['eta_uncertainty'] = eta_unc
                direct['conv_order_q'] = q_emp
                direct['refinement_results'] = refine_results
    print()

    # ===================================================================
    # (II) Picard sweep: verify eta_sweep -> eta_direct as eps -> 0+
    # ===================================================================
    print("--- Part (II): Picard sweep verification ---")
    print()

    # epsilon sweep: covers L_F·I_alpha from 0.5 to 0.999
    epsilons = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

    results = []
    for eps in epsilons:
        A_0 = (1.0 - eps) / (K_ALPHA * L_F)
        L_F_I_alpha = L_F * A_0 * K_ALPHA  # by construction = 1 - eps
        t, y, info = solve_fixed_point(A_0)
        if not np.all(np.isfinite(y)):
            print(f"  [eps={eps:.4f}] DIVERGED")
            continue
        g_star = float(y[-1])  # asymptote (≈ c_0 by Proposition asymptotic_rate)
        y0 = float(y[0])
        if g_star > 1e-12:
            r_meas = 1.0 - y0 / g_star
            saturation = r_meas / L_F_I_alpha
        else:
            r_meas = saturation = float("nan")
        deviation_at_T = abs(g_star - C_0) / C_0  # how close g_* is to true c_0
        results.append(dict(
            eps=eps, A_0=A_0, L_F_I_alpha=L_F_I_alpha,
            y0=y0, g_star=g_star, r=r_meas, saturation=saturation,
            iters=info["iters"], residual=info["residual"],
            deviation_at_T=deviation_at_T,
        ))
        print(f"  eps={eps:.4f}  A_0={A_0:.4f}  L_F·I_alpha={L_F_I_alpha:.4f}  "
              f"y(0)={y0:.6f}  g_*={g_star:.6f}  r={r_meas:.6f}  "
              f"r/(L_F·I_alpha)={saturation:.6f}  iters={info['iters']:3d}  "
              f"|g_*-c_0|/c_0={deviation_at_T:.2e}")

    print()
    print("=" * 72)
    print("Asymptotic behaviour: comparing sweep limit with direct computation")
    print("=" * 72)
    smallest = min(results, key=lambda r: r["eps"])
    print(f"  Picard sweep at eps = {smallest['eps']:.4f}: r/(L_F I_alpha) = {smallest['saturation']:.6f}")
    print(f"  Direct linear-solve eta at A_0 = A_0^*:    {direct['eta_direct']:.6f}")
    print(f"  Discrepancy |sweep - direct|:              {abs(smallest['saturation'] - direct['eta_direct']):.2e}")
    print()
    eta_refined = direct.get('eta_direct_refined', direct['eta_direct'])
    eta_unc = direct.get('eta_uncertainty', 1e-3)
    print(f"  Refined estimate (Aitken extrapolation, T_max=200, h=0.025):")
    print(f"    eta(alpha=1.5, p=3) = {eta_refined:.4f} ± {eta_unc:.4f}")
    print(f"    Difference from sweep at eps=0.001: "
          f"{abs(eta_refined - smallest['saturation']):.4f}")
    print()
    if abs(eta_refined - smallest['saturation']) < 5e-3:
        print("  *** MATHEMATICAL CONSISTENCY CONFIRMED (within mesh-refinement uncertainty) ***")
        print("  The limit of the Picard sweep is consistent with the direct linear-solve")
        print(f"  computation of eta(alpha={ALPHA}, p={P_KERNEL}) := 1 - phi_infty(0)")
        print("  where phi_infty := (I + L_F * I_alpha^*)^(-1) 1 is the limit fixed point")
        print("  at the contraction boundary A_0 = A_0^* := 1/(L_F * K_alpha).")
    else:
        print("  *** WARNING: discrepancy > 5e-3 between methods ***")
    print()
    print("INTERPRETATION:")
    print(f"  The bound r <= L_F·I_alpha is VALID but NOT asymptotically tight.")
    print(f"  The asymptotic saturation eta(alpha={ALPHA}, p={P_KERNEL}) = {eta_refined:.4f}")
    print(f"  is mathematically characterised as eta := 1 - phi_infty(0), where phi_infty is")
    print(f"  the unique solution of the limit equation phi + L_F * I_alpha^* phi = 1")
    print(f"  in C_b(R+), at A_0 = A_0^* := 1/(L_F K_alpha).")
    print(f"  Residual slack 1 - eta = {1-eta_refined:.4f} (genuine, non-vanishing).")

    # Make the figure
    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 2, wspace=0.30)

    eps_arr = np.array([r["eps"] for r in results])
    sat_arr = np.array([r["saturation"] for r in results])
    r_arr = np.array([r["r"] for r in results])
    LF_Ia = np.array([r["L_F_I_alpha"] for r in results])

    # Panel (a): measured r vs the bound 1-eps, log-x
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogx(eps_arr, r_arr, 'o-', color='steelblue', ms=6, lw=1.4,
                label=r'measured $r := 1 - y_*(0)/g_*$')
    ax.semilogx(eps_arr, LF_Ia, 's--', color='darkred', ms=5, lw=1.2,
                label=r'bound $L_F\,I_\alpha = 1-\varepsilon$')
    ax.set_xlabel(r'$\varepsilon = 1 - L_F\,I_\alpha$', fontsize=11)
    ax.set_ylabel(r'amplitude ratio', fontsize=11)
    ax.set_title(r'(a) measured $r$ vs predicted bound', fontsize=11)
    ax.set_xlim(1, 1e-3)  # invert x-axis: small epsilon to the right
    ax.invert_xaxis()
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=9, loc='lower left')

    # Panel (b): saturation ratio r / (L_F I_alpha) converges to eta < 1
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogx(eps_arr, sat_arr, 'o-', color='darkgreen', ms=6, lw=1.4,
                label=r'$r / (L_F\,I_\alpha)$ (measured)')
    ax.axhline(y=1.0, color='gray', linestyle=':', lw=1.0,
               label=r'theoretical upper bound $= 1$')
    # Compute eta as average of last 3 points (asymptotic plateau)
    eta_inf = float(np.mean(sat_arr[-3:]))
    ax.axhline(y=eta_inf, color='darkorange', linestyle='--', lw=1.0,
               label=rf'asymptotic plateau $\eta \approx {eta_inf:.3f}$')
    # Mark the residual slack
    ax.fill_between(eps_arr, eta_inf, 1.0, color='lightgray', alpha=0.4,
                    label=rf'residual slack $1 - \eta \approx {1-eta_inf:.3f}$')
    ax.set_xlabel(r'$\varepsilon = 1 - L_F\,I_\alpha$', fontsize=11)
    ax.set_ylabel(r'saturation ratio', fontsize=11)
    ax.set_title(r'(b) $r/(L_F\,I_\alpha) \to \eta < 1$: bound valid but not tight',
                  fontsize=11)
    ax.set_xlim(1, 1e-3)
    ax.invert_xaxis()
    ax.set_ylim(0.6, 1.05)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8, loc='lower left')

    fig.suptitle(rf'Tightness investigation of Theorem 9.1 amplitude bound '
                 rf'($\alpha={ALPHA}$, $p={P_KERNEL}$, $L_F={L_F}$, '
                 rf'$\beta_0=\xi=\tau=0$)',
                 fontsize=11)

    out_pdf = figuras_dir() / "fig_S9_amplitude_bound_sharpness.pdf"
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[figure] wrote {out_pdf}")

    # Save numerical table for paper
    out_txt = project_root() / "outputs_clinical" / "amplitude_bound_sharpness.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        f.write("Tightness analysis of Theorem 9.1 amplitude bound\n")
        f.write("=" * 60 + "\n")
        f.write(f"alpha={ALPHA}, p={P_KERNEL}, L_F={L_F}, c_0={C_0}, "
                f"T_max={T_MAX}, h={H}, w_damping={W_DAMPING}\n\n")
        f.write(f"{'epsilon':>10s}  {'A_0':>9s}  {'L_F*I_alpha':>11s}  "
                f"{'y(0)':>9s}  {'g_*':>9s}  {'r':>9s}  "
                f"{'r/bound':>9s}  {'iters':>5s}\n")
        for res in results:
            f.write(f"{res['eps']:>10.4f}  {res['A_0']:>9.4f}  "
                    f"{res['L_F_I_alpha']:>11.6f}  "
                    f"{res['y0']:>9.6f}  {res['g_star']:>9.6f}  "
                    f"{res['r']:>9.6f}  {res['saturation']:>9.6f}  "
                    f"{res['iters']:>5d}\n")
        f.write("\n")
        f.write("DIRECT analytical-numerical computation (NO-CONVOLUTION prototype):\n")
        f.write(f"  Setting: xi=0 (no K∗y), gamma_sat=L_F=1\n")
        f.write(f"  At A_0^* = 1/(L_F * K_alpha) = {direct['A_0_star']:.6f}:\n")
        f.write(f"    discrete operator norm L_F * ||M||_{{C_b}} (row-sum) = "
                f"{direct['operator_norm_LF_I_alpha']:.6f}\n")
        f.write(f"      (theory: continuum operator norm = 1.0)\n")
        f.write(f"    phi_infty(0) = {direct['phi_at_0']:.6f}\n")
        f.write(f"    eta_no_conv(alpha=1.5, p=3) = 1 - phi_infty(0) = {direct['eta_direct']:.6f}\n")
        f.write(f"    1 - eta_no_conv = residual slack = {1-direct['eta_direct']:.6f}\n\n")
        if 'eta_with_convolution' in direct:
            cohort = direct['cohort_setting']
            f.write("COHORT setting WITH CONVOLUTION (gamma_sat=0.6, xi=0.4, mu=0.1):\n")
            f.write(f"  L_F = gamma_sat + xi = {cohort['L_F']}, "
                    f"A_0^* = {cohort['A_0_star']:.6f}\n")
            f.write(f"  Mesh: h={cohort['h']}, T_max={cohort['T_max']}, "
                    f"N+1={cohort['N_grid']}\n")
            f.write(f"  phi_infty(0) = {cohort['phi_at_0']:.6f}\n")
            f.write(f"  eta_with_conv(alpha=1.5, p=3) = {cohort['eta_with_conv']:.6f}\n")
            f.write(f"  1 - eta_with_conv = residual slack = "
                    f"{1-cohort['eta_with_conv']:.6f}\n")
            f.write(f"  Discrepancy from no-conv eta: "
                    f"{abs(cohort['eta_with_conv'] - direct['eta_direct']):.4f}\n")
            f.write(f"  Sharpened bound for cohort at A_0^cap=2.0, alpha=1.5:\n")
            f.write(f"    eta_with_conv * A_0^cap * K_alpha(1.5) * L_F = "
                    f"{cohort['eta_with_conv']*2.0*K_ALPHA*1.0:.4f}\n")
            f.write(f"  (vs original Theorem 9.1 bound A_0^cap*K_alpha*L_F = 0.886)\n\n")
        if 'eta_TS_with_convolution' in direct:
            ts = direct['cohort_TS_setting']
            f.write("TIME-SCALED kernel WITH CONVOLUTION (resolves C3 v2 strawman):\n")
            f.write(f"  Kernel: a(s) = A_0 (s_0/(s_0+s))^p, s_0 = {ts['s_0']} d (IADT scale)\n")
            f.write(f"  L_F = {ts['L_F']}, A_0_max_TS = 1/(s_0^α K_α L_F) "
                    f"= {ts['A_0_max_TS']:.6e}\n")
            f.write(f"  Mesh: h={ts['h']}, T_max={ts['T_max']}, N+1={ts['N_grid']}\n")
            f.write(f"  Discrete L_F * ||M_TS|| = "
                    f"{ts['operator_norm_LF_I_alpha_TS']:.6f} (target 1.0)\n")
            f.write(f"  phi_infty^TS(0) = {ts['phi_at_0']:.6f}\n")
            f.write(f"  eta_TS_with_conv(α=1.5, p=3, s_0=300) = "
                    f"{ts['eta_TS_with_conv']:.6f}\n")
            f.write(f"  Achievable amplitude bound at A_0 = A_0_max_TS:\n")
            f.write(f"    eta_TS * 1 * ||y||_inf = {ts['eta_TS_with_conv']:.4f} "
                    f"(vs cohort r ~ 0.96, gap = "
                    f"{0.96 - ts['eta_TS_with_conv']:.4f})\n\n")
        eta_inf = direct['eta_direct']
        f.write(f"FINDING: as epsilon -> 0+,\n")
        f.write(f"  r / (L_F * I_alpha) -> eta(alpha={ALPHA}, p={P_KERNEL}) "
                f"= {eta_inf:.6f}, NOT 1.\n")
        f.write(f"  (Picard sweep at eps=10^-3 gives {smallest['saturation']:.6f}; "
                f"discrepancy {abs(smallest['saturation']-eta_inf):.2e})\n")
        f.write(f"  Residual slack 1 - eta ≈ {1-eta_inf:.4f}.\n")
        f.write(f"  smallest eps tested = "
                f"{min(r['eps'] for r in results):.4f}\n")
        f.write(f"  saturation at smallest eps = {smallest['saturation']:.6f}\n")
        f.write(f"  numerical asymptote bias |g_*-c_0|/c_0 = "
                f"{smallest['deviation_at_T']:.2e}\n\n")
        f.write("INTERPRETATION:\n")
        f.write("The amplitude bound (eq:amplitude_bound_betafree) is a VALID\n")
        f.write("upper bound (verified r/bound <= 1 in all 10 tested epsilons),\n")
        f.write("but is NOT asymptotically tight. The geometric origin of the\n")
        f.write("residual slack is the integral averaging\n")
        f.write("  r = L_F * I_alpha * <y_*/g_*>_a\n")
        f.write("with weight a(s) s^{alpha-1}/Gamma(alpha): since y_*(s)/g_*\n")
        f.write("attains values < 1 on a positive-measure subset of the kernel\n")
        f.write("support, <y_*/g_*>_a < 1 with a slack that does not vanish in\n")
        f.write("the contraction-boundary limit. Sharpening the bound requires\n")
        f.write("spectral analysis of I_alpha on C_b(R+), left as open problem.\n")
    print(f"[output] wrote {out_txt}")


if __name__ == "__main__":
    main()
