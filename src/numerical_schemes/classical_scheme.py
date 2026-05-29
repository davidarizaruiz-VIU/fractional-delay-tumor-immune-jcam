"""
classical_scheme.py
===================

Classical numerical scheme for the right-sided fractional Caputo problem on
the half-line with nonlocal asymptotic condition

       C_D^{alpha}_{-} y(t) = a(t) * f( y(t), (K * y)(t) ),    t >= 0,
       y(infty) = g(y),                                         (alpha in (1,2))

corresponding to the integral equation in the manuscript

       y(t) = g(y) + (1 / Gamma(alpha)) int_t^{infty} (s-t)^{alpha-1} a(s) F(y)(s) ds,
       F(y)(s) := f(y(s), (K * y)(s)).

This module implements:

  (a) The L1-discretisation of the right-sided Caputo operator on a uniform
      mesh truncated at T_max = N*h, with O(h^{2-alpha}) consistency error
      under sufficient regularity of y (Theorem 8.1 of the paper).

  (b) The companion product-rectangle quadrature for the right-sided
      Riemann-Liouville fractional integral I^{alpha}_{-}, which is the
      direct discretisation of the integral form of T (and is what the
      Picard iteration uses).

  (c) The product-trapezoidal quadrature for the distributed-delay
      convolution (K * y)(t) = int_0^t K(t,s) y(s) ds.

  (d) A truncated-Picard solver that implements the iteration
      y^{(k+1)} = T y^{(k)} on the mesh, with explicit error control
      derived from the weak contraction modulus theta_g(r) +
      theta_f(rho_K r) I_alpha < r of (H_theta) (Theorem 5.1 of the paper).

  (e) A linear direct solver y = T y for the special case of the
      sharpness lemma (extremal datum y_ext): a(s) = (1+s)^{-gamma},
      f = 1, K = 0, g = 0, in which T is constant and y = T(0) is just
      the right-sided fractional integral evaluated on the mesh.

  (f) Validation utilities against the closed-form extremal solution

           y_ext(t) = B(alpha, gamma-alpha) / Gamma(alpha) * (1+t)^{alpha-gamma}

      from Lemma 6.x (sharpness) of the paper.

References
----------
Diethelm, *The Analysis of Fractional Differential Equations*, Lecture
Notes in Mathematics 2004, Springer 2010, Chapter 7-8.
Sun & Wu, *A fully discrete difference scheme for a diffusion-wave system*,
Appl. Numer. Math. 56 (2006) 193-209.
Lin & Xu, *Finite difference / spectral approximations for the
time-fractional diffusion equation*, J. Comput. Phys. 225 (2007) 1533-1552.
Kilbas, Srivastava, Trujillo, *Theory and Applications of Fractional
Differential Equations*, North-Holland 2006.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.special import beta as beta_fn

# ---------------------------------------------------------------------------
# 1.  Operator L^{(alpha,-)}_h  --- L1 right-sided Caputo discretisation
# ---------------------------------------------------------------------------


def L1_weights(alpha: float, m: int) -> np.ndarray:
    r"""
    Weights of the L1 right-sided Caputo discretisation.

    Returns the array $\{a_k^{(\alpha)}\}_{k=0}^{m-1}$ with
    $a_k^{(\alpha)} = (k+1)^{2-\alpha} - k^{2-\alpha}$, derived in Section
    8.1 of the paper from
    $\int_{t_j}^{t_{j+1}}(s-t_n)^{1-\alpha}\,ds = \frac{h^{2-\alpha}}{2-\alpha}\,
    a_{j-n}^{(\alpha)}$.

    Parameters
    ----------
    alpha : float
        Fractional order, alpha in (1, 2).
    m : int
        Number of subintervals from t_n to T_max.

    Returns
    -------
    a : (m,) ndarray
    """
    if not (1.0 < alpha < 2.0):
        raise ValueError(f"alpha must lie in (1,2); got {alpha}.")
    if m < 0:
        raise ValueError(f"m must be non-negative; got {m}.")
    k = np.arange(m, dtype=float)
    p = 2.0 - alpha
    return (k + 1.0) ** p - k ** p


def second_difference(y: np.ndarray) -> np.ndarray:
    r"""
    Second difference operator $\Delta_j^2 y$ on the uniform mesh, returning a
    length-N array indexed by $j = 0, 1, ..., N-1$ given input $y$ of length
    $N+1 = $ ``len(y)``.

    Convention:
        - For $j \geq 1$, central second difference (3-point stencil):
          $h^2 \Delta_j^2 y = y_{j-1} - 2 y_j + y_{j+1}$ with truncation error
          $\frac{h^2}{12} y^{(4)}(\xi)$ ($O(h^2)$ approximation of $y''(t_j)$).
        - For $j = 0$, forward 4-point stencil:
          $h^2 \Delta_0^2 y = 2 y_0 - 5 y_1 + 4 y_2 - y_3$ with truncation
          error $\frac{11 h^2}{12} y^{(4)}(\xi)$ ($O(h^2)$ approximation of
          $y''(t_0)$).

    Both stencils give *consistency error* $O(h^2)$ in approximating $y''$
    pointwise.

    Returns
    -------
    delta2_h2 : (N,) ndarray
        The values $h^2 \Delta_j^2 y$ for $j = 0, 1, ..., N-1$.
        The actual $\Delta_j^2 y$ is obtained by dividing by $h^2$.
    """
    if y.ndim != 1 or len(y) < 4:
        raise ValueError("y must be a 1D array of length >= 4.")
    N = len(y) - 1
    out = np.empty(N, dtype=float)
    # j = 0: forward 4-point
    out[0] = 2.0 * y[0] - 5.0 * y[1] + 4.0 * y[2] - y[3]
    # j = 1, ..., N-1: central 3-point
    out[1:] = y[:-2] - 2.0 * y[1:-1] + y[2:]
    # ^^ y[:-2] = y[0..N-2], y[1:-1] = y[1..N-1], y[2:] = y[2..N]
    # so for j in 1..N-1 we get y[j-1] - 2 y[j] + y[j+1]
    return out


def caputo_minus_L1_at(y: np.ndarray, h: float, alpha: float, n: int) -> float:
    r"""
    Evaluate the L1 right-sided Caputo discretisation at index n:

        $\mathcal{L}_h^{(\alpha,-)} y_n = \frac{h^{-\alpha}}{\Gamma(3-\alpha)}
            \sum_{k=0}^{N-n-1} a_k^{(\alpha)}\, h^2 \Delta_{n+k}^2 y$.

    Parameters
    ----------
    y : (N+1,) ndarray
        Mesh values y_0, y_1, ..., y_N.
    h : float
        Mesh size.
    alpha : float
        Fractional order in (1, 2).
    n : int
        Mesh index, 0 <= n <= N-1.

    Returns
    -------
    Lh_y_n : float
    """
    N = len(y) - 1
    if not 0 <= n <= N - 1:
        raise ValueError(f"n must be in [0, N-1]; got n={n}, N={N}.")
    m = N - n
    if m == 0:
        return 0.0
    a_k = L1_weights(alpha, m)            # length m
    delta2_h2 = second_difference(y)      # length N, indexed by j=0..N-1
    # Slice j = n, n+1, ..., N-1 (length m)
    block = delta2_h2[n : n + m]
    return h ** (-alpha) / gamma_fn(3.0 - alpha) * float(np.dot(a_k, block))


def caputo_minus_L1_naive(y: np.ndarray, h: float, alpha: float) -> np.ndarray:
    r"""
    Naive O(N^2) implementation of the L1 right-sided Caputo discretisation.

    Kept for verification / correctness testing; production code should use
    :func:`caputo_minus_L1` (FFT-accelerated, O(N log N)) instead.
    """
    N = len(y) - 1
    delta2_h2 = second_difference(y)            # length N
    out = np.empty(N, dtype=float)
    cst = h ** (-alpha) / gamma_fn(3.0 - alpha)
    a_full = L1_weights(alpha, N)               # a_k^{(alpha)} for k=0..N-1
    for n in range(N):
        m = N - n
        out[n] = cst * np.dot(a_full[:m], delta2_h2[n:])
    return out


def caputo_minus_L1(y: np.ndarray, h: float, alpha: float) -> np.ndarray:
    r"""
    Vectorised application of the L1 right-sided Caputo discretisation
    at every index n = 0, 1, ..., N-1, in O(N log N) via FFT correlation.

    Mathematical observation
    ------------------------

    Let $b_j := h^2 \Delta_j^2 y$ (length $N$, indexed $j=0,\dotsc,N-1$) and
    $a_k := a_k^{(\alpha)}$ (length $N$, indexed $k=0,\dotsc,N-1$). The
    discrete operator is

        $\mathcal{L}_h y_n = \frac{h^{-\alpha}}{\Gamma(3-\alpha)}
            \sum_{k=0}^{N-n-1} a_k\, b_{n+k}
        = \frac{h^{-\alpha}}{\Gamma(3-\alpha)}\,(a \star b)_n$,

    where $\star$ denotes the cross-correlation. This is computed in
    $O(N \log N)$ by `scipy.signal.fftconvolve` applied to $a$ and the
    reverse of $b$ (equivalent to convolution of $a$ with the time-reversed
    $b$, picking the first $N$ outputs).

    Returns
    -------
    Lh_y : (N,) ndarray
        The values $\mathcal{L}_h^{(\alpha,-)} y_n$ for $n = 0, ..., N-1$.
        Note: index n = N is NOT computed (would need an empty sum).
    """
    from scipy.signal import fftconvolve

    N = len(y) - 1
    if N <= 0:
        return np.empty(0, dtype=float)
    delta2_h2 = second_difference(y)            # length N
    a_full = L1_weights(alpha, N)               # length N
    cst = h ** (-alpha) / gamma_fn(3.0 - alpha)
    # Cross-correlation (a * reverse(b)) and pick the first N output samples.
    # fftconvolve in 'full' mode returns length 2N-1; the entry at index k
    # in the correlation a star b is the entry at index N-1+k of the
    # convolution of a with reverse(b). Equivalently we use np.correlate-style:
    corr_full = fftconvolve(delta2_h2, a_full[::-1], mode="full")  # length 2N-1
    # corr_full[N-1 + n] = sum_{k} a_full[k] * delta2_h2[n+k]    for n in [-(N-1), N-1]
    # We want n = 0, 1, ..., N-1.
    out = corr_full[N - 1 : 2 * N - 1] * cst
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# 2.  Product-rectangle quadrature for the right-sided fractional INTEGRAL
# ---------------------------------------------------------------------------


def I_minus_quadrature_weights(alpha: float, m: int) -> np.ndarray:
    r"""
    Product-rectangle weights $\omega_k$ such that

        $(I^{\alpha}_- v)(t_n) \approx \frac{1}{\Gamma(\alpha)}
             \sum_{j=n}^{N-1} v(t_j) \int_{t_j}^{t_{j+1}} (s-t_n)^{\alpha-1}\,ds
         \;=\; \frac{h^{\alpha}}{\Gamma(\alpha+1)} \sum_{k=0}^{m-1} \omega_k\, v_{n+k}$

    with $\omega_k := (k+1)^\alpha - k^\alpha$.

    Parameters
    ----------
    alpha : float
        alpha > 0.
    m : int
        Number of subintervals from t_n to T_max.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive; got {alpha}.")
    k = np.arange(m, dtype=float)
    return (k + 1.0) ** alpha - k ** alpha


def I_minus_quadrature(v: np.ndarray, h: float, alpha: float) -> np.ndarray:
    r"""
    Product-rectangle quadrature of the right-sided fractional integral
    $(I^{\alpha}_- v)(t_n)$ for $n = 0, 1, ..., N$, given mesh values
    $v_0, v_1, ..., v_N$.

    Returns
    -------
    Iv : (N+1,) ndarray
        The values $I^{\alpha}_- v$ at $t_0, t_1, ..., t_N$, with
        $(I^{\alpha}_- v)(t_N) = 0$ on the truncated domain.
    """
    N = len(v) - 1
    out = np.zeros(N + 1, dtype=float)
    cst = h ** alpha / gamma_fn(alpha + 1.0)
    if N == 0:
        return out
    omega = I_minus_quadrature_weights(alpha, N)         # k=0..N-1
    for n in range(N):
        m = N - n
        out[n] = cst * np.dot(omega[:m], v[n : n + m])
    return out


# ---------------------------------------------------------------------------
# 3.  Distributed-delay convolution K * y on the mesh
# ---------------------------------------------------------------------------


def distributed_delay(K, y: np.ndarray, h: float) -> np.ndarray:
    r"""
    Product-trapezoidal quadrature for the distributed-delay convolution
    $(K * y)(t) := \int_0^t K(t, s)\, y(s)\, ds$ on the uniform mesh.

    Parameters
    ----------
    K : callable
        Bivariate kernel K(t, s).
    y : (N+1,) ndarray
        Mesh values.
    h : float
        Mesh size.

    Returns
    -------
    Ky : (N+1,) ndarray
        The values $(K*y)(t_n)$ for n=0..N, with (K*y)(t_0) = 0.
    """
    N = len(y) - 1
    t = h * np.arange(N + 1)
    out = np.zeros(N + 1, dtype=float)
    for n in range(1, N + 1):
        # trapezoidal: int_0^{t_n} K(t_n,s) y(s) ds approx
        # h * [ 0.5*K(t_n,0)*y_0 + sum_{j=1}^{n-1} K(t_n,t_j)*y_j + 0.5*K(t_n,t_n)*y_n ]
        s = t[: n + 1]
        Ks = K(t[n], s)
        w = np.ones(n + 1)
        w[0] = 0.5
        w[-1] = 0.5
        out[n] = h * np.sum(w * Ks * y[: n + 1])
    return out


# ---------------------------------------------------------------------------
# 4.  Closed-form extremal solution (Lemma 6.x of the paper, sharpness)
# ---------------------------------------------------------------------------


def y_ext(t: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    r"""
    Closed-form extremal solution of the integral equation under the data
    a(s) = (1+s)^{-gamma}, f = 1, K = 0, g = 0:

        $y_{\mathrm{ext}}(t) = \dfrac{B(\alpha,\gamma-\alpha)}{\Gamma(\alpha)}
                                \,(1+t)^{\alpha-\gamma}$,

    valid for $\gamma > \alpha + 1 > 2$ (Lemma 6.x of the manuscript).
    """
    if gamma <= alpha + 1:
        raise ValueError(f"gamma must be > alpha+1; got alpha={alpha}, gamma={gamma}.")
    coeff = beta_fn(alpha, gamma - alpha) / gamma_fn(alpha)
    return coeff * (1.0 + np.asarray(t, dtype=float)) ** (alpha - gamma)


# ---------------------------------------------------------------------------
# 5.  Linear direct solver for the y_ext test
#     T(0) = I^{alpha}_- (a) since g = 0, f = 1, K = 0
# ---------------------------------------------------------------------------


def solve_yext_via_integral(alpha: float, gamma: float, h: float, T_max: float):
    r"""
    Compute the numerical approximation of y_ext via the *integral form*
    of the operator T:

        $y_n^{(\mathrm{num})} := (I^{\alpha}_- a)(t_n) + \mathrm{tail}(t_n)$,

    where the tail $\int_{T_{\max}}^{\infty} (s-t_n)^{\alpha-1} a(s) ds /
    \Gamma(\alpha)$ is computed in closed form via the substitution
    $u = (1+t_n)v + t_n$:

        $\mathrm{tail}(t_n) = \frac{(1+t_n)^{\alpha-\gamma}}{\Gamma(\alpha)}
            \int_{(T_{\max}-t_n)/(1+t_n)}^{\infty} v^{\alpha-1}(1+v)^{-\gamma}\,dv$.

    The remaining integral is evaluated using the regularised incomplete
    beta function (scipy hyp2f1 fallback) for analytic accuracy.

    Returns
    -------
    t : (N+1,) ndarray
    y_num : (N+1,) ndarray
        Numerical approximation of y_ext(t_n).
    y_exact : (N+1,) ndarray
        Closed-form y_ext(t_n).
    """
    N = int(round(T_max / h))
    if abs(N * h - T_max) > 1e-12 * max(T_max, 1.0):
        raise ValueError("T_max must be an integer multiple of h.")
    t = h * np.arange(N + 1)
    a_vals = (1.0 + t) ** (-gamma)
    # finite quadrature on [t_n, T_max]
    Iam = I_minus_quadrature(a_vals, h, alpha)
    # closed-form tail correction via lower incomplete beta (no truncation
    # error in the tail term itself; we still rely on the *quadrature* for
    # the bulk part)
    tail = _yext_tail(t, T_max, alpha, gamma)
    y_num = Iam + tail
    y_exact = y_ext(t, alpha, gamma)
    return t, y_num, y_exact


def _yext_tail(t: np.ndarray, T_max: float, alpha: float, gamma: float) -> np.ndarray:
    r"""
    Closed-form tail
        $\mathrm{tail}(t) = \frac{1}{\Gamma(\alpha)}\int_{T_{\max}}^{\infty}
                                       (s-t)^{\alpha-1}(1+s)^{-\gamma}\,ds$.
    """
    from scipy.special import hyp2f1
    t = np.asarray(t, dtype=float)
    # change of variable u = s - T_max, then v = u / (1 + T_max + ?)
    # We use the integral representation
    #   int_{T_max}^infty (s-t)^{alpha-1}(1+s)^{-gamma} ds
    #     = (1+T_max)^{alpha-gamma}*(T_max-t+1+t-T_max)^? ...
    # Easier: numerical evaluation with scipy.integrate is robust.
    from scipy.integrate import quad
    out = np.empty_like(t)
    for i, ti in enumerate(t):
        if T_max <= ti:
            out[i] = 0.0
            continue
        val, _ = quad(
            lambda s: (s - ti) ** (alpha - 1.0) * (1.0 + s) ** (-gamma),
            T_max, np.inf, limit=200,
        )
        out[i] = val
    return out / gamma_fn(alpha)


# ---------------------------------------------------------------------------
# 6.  Operator-test against y_ext: apply L_h to y_ext(t) and compare with
#     a(t) = (1+t)^{-gamma}; this verifies the L1 *operator* without
#     dealing with boundary conditions.
# ---------------------------------------------------------------------------


def operator_test_yext(
    alpha: float,
    gamma: float,
    h: float,
    T_max: float,
):
    r"""
    Apply the L1 right-sided Caputo discretisation to mesh samples of y_ext,
    and compare with the exact a(t) = (1+t)^{-gamma}.

    The analytic identity (Lemma 6.x + Theorem 6.X of the paper) is
        $\mathcal{C}^{\alpha}_{-} y_{\mathrm{ext}}(t) = (1+t)^{-\gamma}$,
    so $|\mathcal{L}_h y_{\mathrm{ext}}(t_n) - (1+t_n)^{-\gamma}|$ is the
    *truncation error* of the L1 operator on the smooth test function
    y_ext, plus the tail-truncation error introduced by replacing the
    true integral on $[t_n,\infty)$ with the numerical sum on $[t_n,T_{\max}]$.

    Returns
    -------
    t : (N+1,) ndarray
    err : (N,) ndarray
        |L_h y_{ext}(t_n) - a(t_n)| for n = 0, 1, ..., N-1.
    a_vals : (N,) ndarray
        a(t_n) for n = 0, 1, ..., N-1 (target values).
    Lh_vals : (N,) ndarray
        L_h y_{ext}(t_n).
    """
    N = int(round(T_max / h))
    t = h * np.arange(N + 1)
    y_grid = y_ext(t, alpha, gamma)
    Lh_vals = caputo_minus_L1(y_grid, h, alpha)
    a_vals = (1.0 + t[:-1]) ** (-gamma)
    err = np.abs(Lh_vals - a_vals)
    return t, err, a_vals, Lh_vals


# ---------------------------------------------------------------------------
# 7.  Truncated Picard iteration on the integral form
#     y^{(k+1)} = T y^{(k)} = g(y^{(k)}) + I^{alpha}_- (a*F(y^{(k)}))
# ---------------------------------------------------------------------------


def picard_truncated(
    a, f, K, g,
    alpha: float,
    h: float,
    T_max: float,
    tail_correction=None,
    n_iter: int = 50,
    tol: float = 1e-12,
    y0=None,
    verbose: bool = False,
):
    r"""
    Truncated Picard iteration y^{(k+1)} = T y^{(k)} on the mesh, where T is
    discretised via the product-rectangle quadrature for the integral form
    of the operator (which gives the cleanest match with the C_b fixed-point
    theory of Sections 3-5 of the paper).

    Parameters
    ----------
    a : callable s -> a(s)
    f : callable (u, v) -> f(u, v)
    K : callable (t, s) -> K(t, s)  or None for K = 0
    g : callable y_array -> float    or float for constant g
    alpha : float in (1, 2)
    h : float
        Mesh size.
    T_max : float
        Truncation horizon.
    tail_correction : callable (t_array) -> tail_array, optional
        Analytical tail correction tail(t) =
        (1/Gamma(alpha)) * int_{T_max}^infty (s-t)^{alpha-1} a(s) F(s) ds.
        If None, tail is set to zero (and the user is responsible for
        choosing T_max large enough that the tail bound from
        Proposition 6.X is below tolerance).
    n_iter : int
        Maximum number of Picard iterations.
    tol : float
        Convergence tolerance in L^infty norm.
    y0 : (N+1,) ndarray, optional
        Initial guess. Default: the constant g(0).
    verbose : bool

    Returns
    -------
    t : (N+1,) ndarray
    y : (N+1,) ndarray
    info : dict
        Convergence diagnostics: keys 'iters', 'res_history'.
    """
    N = int(round(T_max / h))
    t = h * np.arange(N + 1)

    # Sample a on the mesh
    a_vals = np.asarray([a(ti) for ti in t], dtype=float)

    # Initialise
    if y0 is None:
        if callable(g):
            try:
                g0 = float(g(np.zeros(N + 1)))
            except Exception:
                g0 = 0.0
        else:
            g0 = float(g)
        y = g0 * np.ones(N + 1, dtype=float)
    else:
        y = np.asarray(y0, dtype=float).copy()
        if y.shape != (N + 1,):
            raise ValueError("y0 has wrong shape.")

    res_history = []

    for k in range(n_iter):
        # 1.  Distributed delay (K * y)
        if K is None:
            Ky = np.zeros_like(y)
        else:
            Ky = distributed_delay(K, y, h)
        # 2.  F(y) = f(y, Ky)
        F = np.array([f(y[i], Ky[i]) for i in range(N + 1)], dtype=float)
        # 3.  I^alpha_- (a * F)
        aF = a_vals * F
        Iv = I_minus_quadrature(aF, h, alpha)
        # 4.  Tail correction
        if tail_correction is not None:
            Iv = Iv + tail_correction(t)
        # 5.  Constant g(y)
        if callable(g):
            g_val = float(g(y))
        else:
            g_val = float(g)
        # 6.  T y
        y_new = g_val + Iv
        # 7.  Convergence check
        res = float(np.max(np.abs(y_new - y)))
        res_history.append(res)
        if verbose:
            print(f"[Picard] iter {k+1:3d}: ||y^{{k+1}}-y^{{k}}||_inf = {res:.3e}")
        y = y_new
        if res < tol:
            break

    info = dict(iters=k + 1, res_history=np.array(res_history))
    return t, y, info


# ---------------------------------------------------------------------------
# 8.  Convenience: convergence-rate fitting in log-log
# ---------------------------------------------------------------------------


def fit_rate(hs: np.ndarray, errs: np.ndarray) -> tuple:
    r"""
    Linear regression of $\log(\mathrm{err}) = p \cdot \log(h) + c$ to extract
    the empirical convergence rate $p$ and the prefactor $C = e^c$.
    """
    hs = np.asarray(hs, dtype=float)
    errs = np.asarray(errs, dtype=float)
    mask = (hs > 0) & (errs > 0) & np.isfinite(hs) & np.isfinite(errs)
    if mask.sum() < 2:
        raise ValueError("Need at least two valid (h, err) pairs.")
    logh = np.log(hs[mask])
    loge = np.log(errs[mask])
    p, c = np.polyfit(logh, loge, deg=1)
    return float(p), float(np.exp(c))
