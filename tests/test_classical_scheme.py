"""
test_classical_scheme.py
========================

Validation tests for the classical numerical scheme of Section 8.1 of the
paper.  All tests use the closed-form extremal solution y_ext from Lemma 6.4
(sharpness) of the manuscript as analytic ground truth.

Run with::

    python3 test_classical_scheme.py

A non-zero exit code signals that some test failed (see assertions below).
"""

from __future__ import annotations

import numpy as np

from classical_scheme import (
    L1_weights,
    I_minus_quadrature_weights,
    second_difference,
    caputo_minus_L1,
    operator_test_yext,
    solve_yext_via_integral,
    picard_truncated,
    y_ext,
    fit_rate,
    distributed_delay,
    _yext_tail,
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_L1_weights_telescope() -> None:
    """The L1 weights telescope to (m)^{2-alpha}."""
    for alpha in [1.2, 1.5, 1.8]:
        m = 50
        a = L1_weights(alpha, m)
        s = float(np.sum(a))
        ref = m ** (2.0 - alpha)
        assert abs(s - ref) < 1e-12, f"alpha={alpha}: sum={s}, ref={ref}"
    print("[OK] L1 weights telescope to m^(2-alpha)")


def test_quadrature_weights_telescope() -> None:
    """The product-rectangle weights telescope to m^alpha."""
    for alpha in [0.3, 1.2, 1.5, 1.8, 2.5]:
        m = 50
        w = I_minus_quadrature_weights(alpha, m)
        s = float(np.sum(w))
        ref = m ** alpha
        assert abs(s - ref) < 1e-10, f"alpha={alpha}: sum={s}, ref={ref}"
    print("[OK] Product-rectangle weights telescope to m^alpha")


def test_second_difference_polynomial() -> None:
    """Second-difference operator reproduces y'' for polynomials of degree <= 3."""
    for h in [0.1, 0.05]:
        N = 20
        t = h * np.arange(N + 1)
        for poly, ddpoly in [
            (lambda x: x, lambda x: 0.0 * x),                       # y=t,   y''=0
            (lambda x: x ** 2, lambda x: 2.0 + 0.0 * x),            # y=t^2, y''=2
            (lambda x: x ** 3, lambda x: 6.0 * x),                  # y=t^3, y''=6 t
        ]:
            y = poly(t)
            d2_h2 = second_difference(y)
            d2 = d2_h2 / h ** 2
            ref = ddpoly(t[:-1])
            err = float(np.max(np.abs(d2 - ref)))
            assert err < 1e-9, f"poly: max err = {err}, h={h}"
    print("[OK] Second-difference reproduces y'' for cubic polynomials")


def test_caputo_minus_L1_consistency_smooth() -> None:
    """
    Apply L1 to smooth y_ext and check that the operator converges to
    a(t) = (1+t)^{-gamma} at *at least* O(h^{2-alpha}) rate.
    """
    T_max = 80.0
    hs = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
    for alpha, gamma in [(1.2, 3.7), (1.5, 4.0), (1.8, 4.3)]:
        errs = []
        for h in hs:
            t, err, _, _ = operator_test_yext(alpha, gamma, h, T_max)
            N = len(t) - 1
            i_lo = max(1, int(0.02 * N))
            i_hi = N - max(8, int(0.05 * N))
            errs.append(float(np.max(np.abs(err[i_lo:i_hi]))))
        p, _ = fit_rate(hs, errs)
        # Theoretical bound: p >= 2 - alpha
        # Empirical: typically p >= 1 for smooth y; we allow margin
        threshold = max(2.0 - alpha - 0.1, 0.1)
        assert p > threshold, (
            f"alpha={alpha}, gamma={gamma}: rate p={p} below threshold {threshold}"
        )
        print(f"[OK] alpha={alpha:.2f}, gamma={gamma:.2f}: empirical rate p={p:.3f} "
              f"(theoretical bound 2-alpha={2-alpha:.2f})")


def test_solver_integral_converges_to_yext() -> None:
    """Direct integral solver matches y_ext analytically as h -> 0."""
    alpha, gamma = 1.5, 3.5
    T_max = 30.0
    hs = [0.4, 0.2, 0.1, 0.05]
    errs = []
    for h in hs:
        t, y_num, y_exact = solve_yext_via_integral(alpha, gamma, h, T_max)
        errs.append(float(np.max(np.abs(y_num - y_exact))))
    p, _ = fit_rate(hs, errs)
    assert p > 0.8, f"Integral solver rate p={p} too low."
    print(f"[OK] Integral solver converges to y_ext at empirical rate p={p:.3f}")


def test_picard_linear_one_step() -> None:
    """Picard on linear problem (f=1, K=0, g=0) converges in <=2 steps."""
    alpha, gamma = 1.5, 3.5
    T_max = 30.0
    h = 0.1

    def a(s):
        return (1.0 + s) ** (-gamma)

    def f(u, v):
        return 1.0

    def tail(t_arr):
        return _yext_tail(t_arr, T_max, alpha, gamma)

    t, y, info = picard_truncated(
        a=a, f=f, K=None, g=0.0,
        alpha=alpha, h=h, T_max=T_max,
        tail_correction=tail,
        n_iter=10, tol=1e-14,
    )
    assert info["iters"] <= 2, f"Picard linear iters = {info['iters']}, expected <=2"
    err = float(np.max(np.abs(y - y_ext(t, alpha, gamma))))
    assert err < 5e-2, f"Picard linear error = {err}"
    print(f"[OK] Picard converges in {info['iters']} iter on linear problem, "
          f"||err||_inf = {err:.3e}")


def test_picard_nonlinear_smoke() -> None:
    """Picard on a nonlinear problem with bounded f converges to fixed point."""
    alpha = 1.5
    gamma = 3.5
    T_max = 20.0
    h = 0.1

    def a(s):
        return (1.0 + s) ** (-gamma)

    def f(u, v):
        return np.sin(u) + 0.1 * v

    def K(t, s):
        return 2.0 * np.exp(-2.0 * np.maximum(t - s, 0.0))

    g_const = 0.1

    t, y, info = picard_truncated(
        a=a, f=f, K=K, g=g_const,
        alpha=alpha, h=h, T_max=T_max,
        tail_correction=None,
        n_iter=200, tol=1e-10,
    )
    assert info["iters"] < 200, "Picard nonlinear failed to converge"
    assert info["res_history"][-1] < 1e-10
    # y(T_max) should be close to g (decay condition)
    assert abs(y[-1] - g_const) < 1e-3, f"y(T_max) = {y[-1]} vs g = {g_const}"
    print(f"[OK] Picard nonlinear: {info['iters']} iters, residual "
          f"{info['res_history'][-1]:.2e}, y(T_max)={y[-1]:.6f} (g={g_const})")


def test_distributed_delay_constant_kernel() -> None:
    """For K(t,s) = 1, (K*y)(t) = int_0^t y(s) ds; check on y(s) = s."""
    h = 0.01
    N = 200
    t = h * np.arange(N + 1)
    y_vals = t.copy()  # y(s) = s
    Ky = distributed_delay(lambda t, s: np.ones_like(s) if hasattr(s, '__len__')
                           else 1.0, y_vals, h)
    Ky_exact = 0.5 * t ** 2  # int_0^t s ds = t^2 / 2
    err = float(np.max(np.abs(Ky - Ky_exact)))
    # Trapezoidal rule has error O(h^2); for y=s exactly, error = 0 in exact arith
    assert err < 1e-12, f"Trapezoidal on K=1, y=s failed: err = {err}"
    print(f"[OK] Distributed delay quadrature exact on (K=1, y=s): err={err:.2e}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("Validation tests for classical_scheme.py")
    print("=" * 72)
    tests = [
        test_L1_weights_telescope,
        test_quadrature_weights_telescope,
        test_second_difference_polynomial,
        test_caputo_minus_L1_consistency_smooth,
        test_solver_integral_converges_to_yext,
        test_picard_linear_one_step,
        test_picard_nonlinear_smoke,
        test_distributed_delay_constant_kernel,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures == 0:
        print("All tests passed.")
        return 0
    print(f"{failures} test(s) FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
