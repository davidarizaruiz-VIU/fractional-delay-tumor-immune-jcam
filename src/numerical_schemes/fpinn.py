"""
fpinn.py
========

Asymptotically-aware fractional Physics-Informed Neural Network (fPINN) for
the right-sided Caputo problem on the half-line with nonlocal asymptotic
condition (Section 8.2 of the paper).

Architecture
------------
The network is

    y_NN(t; theta) = G(y_NN; c_0, tau, T) + (1+t)^{alpha - p} * N_theta(t),

with G(y) = c_0 + tau * T^{-1} * int_0^T y(s) ds  (the asymptotic functional
g of Section 7 of the paper) and N_theta a small MLP with tanh activation.
This hard-codes architecturally

  (i)   the nonlocal asymptotic condition y(infty) = G(y), since
        (1+t)^{alpha-p} -> 0 as t -> infty when p > alpha;
  (ii)  the polynomial decay y(t) - G(y) ~ (1+t)^{alpha-p} of
        Proposition 6.2 of the paper;
  (iii) compatibility with the integral form y = T y of the abstract
        theory of Sections 3-5.

The Caputo right-sided residual is computed via the same L1
discretisation of Section 8.1 of the paper, evaluated on a fixed mesh
{t_n}_{n=0}^N. This keeps the fPINN consistent with the classical
ground-truth scheme: the discrete operator that defines `correctness'
of the network is exactly the one whose convergence has been proved
in Theorem 8.1 (thm:L1_consistency).

Loss functional
---------------

    L(theta, params) = lambda_R * (1/|I_h|) * sum_{n in I_h}
                              | L_h y_NN(t_n) - a(t_n) F(y_NN)(t_n) |^2
                     + lambda_D * (1/m) * sum_j | y_NN(s_j) - y_obs(s_j) |^2
                     + lambda_0 * | y_NN(0) - 1 |^2,

where I_h = {0, 1, ..., floor(N/2)} is the interior index set on which
Theorem 8.1 of the manuscript provides uniform control of the L1
consistency error. Restricting the residual to I_h aligns the
training functional with the régime in which the variational
consistency theorem (Theorem 8.2) is rigorous, so that the
algorithm minimises exactly the loss covered by the analytical
result.

The third term (calibration constraint y(0) = 1) is optional. The
nonlocal-asymptotic penalty lambda_NL of plain PINN formulations is
ABSENT: the architecture enforces it exactly, not by penalty.

Forward and inverse modes
-------------------------
- Forward: parameters (alpha, A_0, p, beta_0, gamma_sat, xi, lambda_K, c_0, tau, T)
  are held fixed, only theta is optimised; lambda_D = 0 (no observation data).
- Inverse: theta is optimised together with the active subset of parameters
  (default 5: alpha, A_0, lambda_K, c_0, tau); lambda_D > 0 with observation data.

NOTE on notation
----------------
The kernel relaxation rate (called mu in earlier drafts) is now denoted
lambda_K in the paper to keep it notationally disjoint from the bootstrap
exponent nu of |y''| introduced in hypothesis (H^{num}) of Section 8.1.
Variable names in this script (e.g. `mu` if any remain) refer to the
kernel relaxation rate lambda_K of equation (11) of the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# Use double precision throughout: the L1 weights involve (k+1)^{2-alpha} - k^{2-alpha}
# which suffers from catastrophic cancellation in float32 for large k.
torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------------------
# 1.  L1 right-sided Caputo operator (PyTorch / differentiable)
# ---------------------------------------------------------------------------


def L1_weights_torch(alpha: torch.Tensor, m: int) -> torch.Tensor:
    r"""
    Return the array $\{a_k^{(\alpha)}\}_{k=0}^{m-1}$ of L1 weights with
    $a_k^{(\alpha)} = (k+1)^{2-\alpha} - k^{2-\alpha}$.

    The case $k=0$ is treated separately to avoid evaluating
    $0^{2-\alpha}$ (which is mathematically zero but produces ill-defined
    backward gradients in $\alpha$ via $\log 0$); we set $a_0 = 1$
    by hand.

    Parameters
    ----------
    alpha : 0-d torch.Tensor
        Fractional order. May require gradient.
    m : int
        Number of weights.

    Returns
    -------
    a : (m,) torch.Tensor
    """
    if m <= 0:
        return torch.empty(0, dtype=alpha.dtype, device=alpha.device)
    p = 2.0 - alpha
    a0 = torch.ones(1, dtype=alpha.dtype, device=alpha.device)
    if m == 1:
        return a0
    k = torch.arange(1, m, dtype=alpha.dtype, device=alpha.device)
    a_rest = (k + 1.0).pow(p) - k.pow(p)
    return torch.cat([a0, a_rest])


def caputo_minus_L1_torch(
    y: torch.Tensor, h: float, alpha: torch.Tensor,
) -> torch.Tensor:
    r"""
    Apply the L1 right-sided Caputo discretisation to the mesh values
    `y` (length $N+1$), returning a tensor of length $N$ with the values
    $\mathcal{L}_h y_n$ for $n = 0, 1, \dots, N-1$.

    Convention identical to the NumPy `classical_scheme.caputo_minus_L1`:
        - $j=0$: forward 4-point stencil, $h^2 \Delta_0^2 = 2y_0 - 5y_1 + 4y_2 - y_3$
        - $1 \leq j \leq N-1$: central 3-point stencil
        - $\mathcal{L}_h y_n = \frac{h^{-\alpha}}{\Gamma(3-\alpha)}
              \sum_{k=0}^{N-n-1} a_k^{(\alpha)} \cdot h^2 \Delta_{n+k}^2$

    All operations are differentiable with respect to both `y` and `alpha`.
    """
    if y.dim() != 1 or y.shape[0] < 4:
        raise ValueError("y must be a 1D tensor of length >= 4.")
    N = y.shape[0] - 1

    # Second-difference vector (length N), indexed by j = 0, 1, ..., N-1
    d0 = 2.0 * y[0] - 5.0 * y[1] + 4.0 * y[2] - y[3]
    d_int = y[:-2] - 2.0 * y[1:-1] + y[2:]
    delta2_h2 = torch.cat([d0.unsqueeze(0), d_int])  # length N

    # Weights a_k for k = 0, ..., N-1
    a_full = L1_weights_torch(alpha, N)

    # Build a Toeplitz upper-triangular matrix W of shape (N, N) with
    # W[n, j] = a_{j-n} if j >= n, else 0; then L_h y = (cst) * W @ delta2_h2.
    # For moderate N (typically 50-400 in our experiments) this is cheap.
    idx_n = torch.arange(N, device=y.device).unsqueeze(1)             # (N, 1)
    idx_j = torch.arange(N, device=y.device).unsqueeze(0)             # (1, N)
    diff = idx_j - idx_n                                              # (N, N)
    mask = diff >= 0
    a_lookup = torch.where(mask, a_full[diff.clamp(min=0)], torch.zeros_like(diff, dtype=a_full.dtype))
    Lh = (a_lookup @ delta2_h2)                                       # (N,)

    # Constant factor h^{-alpha} / Gamma(3-alpha)
    cst = h ** (-alpha) / torch.exp(torch.lgamma(3.0 - alpha))
    return cst * Lh


def distributed_delay_torch(
    K_func, y: torch.Tensor, t: torch.Tensor, h: float,
) -> torch.Tensor:
    r"""
    Product-trapezoidal quadrature for the convolution
    $(K \ast y)(t) = \int_0^t K(t,s)\,y(s)\,ds$ on the uniform mesh,
    in PyTorch. Returns a tensor of length $N+1$.
    """
    N = y.shape[0] - 1
    out_list = [torch.zeros((), dtype=y.dtype, device=y.device)]
    for n in range(1, N + 1):
        s = t[: n + 1]
        Ks = K_func(t[n], s)
        w = torch.ones(n + 1, dtype=y.dtype, device=y.device)
        w[0] = 0.5
        w[-1] = 0.5
        out_list.append(h * torch.sum(w * Ks * y[: n + 1]))
    return torch.stack(out_list)


# ---------------------------------------------------------------------------
# 2.  MLP and asymptotically-aware wrapper
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    r"""
    Feed-forward MLP with tanh activations: t -> N_theta(t).

    The tanh activation is globally Lipschitz (constant 1), which is
    relevant for the modulus-of-continuity estimates needed in the
    structural identifiability theorem (Corollary 8.X).
    """

    def __init__(self, hidden_layers: Sequence[int] = (32, 32, 32, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = 1
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (N+1,) -> shape (N+1, 1) -> output (N+1, 1) -> squeeze to (N+1,)
        if t.dim() == 1:
            t_in = t.unsqueeze(-1)
        else:
            t_in = t
        return self.net(t_in).squeeze(-1)


class AsymptoticallyAwareNet(nn.Module):
    r"""
    Wrapper that codifies architecturally the nonlocal condition
    $y(\infty) = G(y)$ and the polynomial decay $(1+t)^{\alpha-p}$ of
    Proposition 6.2. The forward map is

        y_NN(t) = G_param + (1+t)^{decay_exp} * N_theta(t),

    with $G_{\mathrm{param}}$ either a learnable constant (when $g$ is
    constant), or the integral functional $c_0 + \tau\,T^{-1}\!\int_0^T
    y_{NN}(s)\,ds$ implemented as a fixed-point iteration on the mesh
    \textup{(}two passes suffice in practice since the mapping is a weak
    contraction in $\tau$\textup{)}, see :py:meth:`forward_with_g`.

    The decay exponent ``decay_exp`` is taken **fixed** in $\alpha$
    \textup{(}at the initial guess $\alpha_0 - p_0$, or any prescribed
    constant\textup{)}: this is essential for identifiability of $\alpha$
    in the inverse problem, because otherwise the network can
    compensate a wrong $\alpha$ by re-scaling $\mathrm{MLP}_\theta$ and
    the inverse residual loses sensitivity to $\alpha$. The teorema of
    asymptotic decay of Proposition 6.2 only requires the exponent to be
    a value $\leq \alpha - p$; using the initial guess preserves the
    architectural decay encoding without coupling to the trainable
    $\alpha$.

    Parameters
    ----------
    alpha, p : torch.Tensor
        Scalar tensors. May require gradient (inverse problem). They
        enter the L1 operator and the rhs $a$, but NOT the architectural
        decay factor (see ``decay_exp``).
    decay_exp : float or 0-d torch.Tensor
        Fixed exponent of the architectural decay $(1+t)^{\mathtt{decay\_exp}}$.
        For the forward-problem sanity check on $y_{\mathrm{ext}}$ we set
        ``decay_exp = alpha_0 - gamma`` with $\alpha_0$ the (known)
        forward $\alpha$. For the inverse problem we set
        ``decay_exp = alpha_0 - p_0`` with $(\alpha_0, p_0)$ the initial
        guess of the optimiser; the learnable $\alpha$ adjusts the L1
        operator only.
    G_const : float or torch.Tensor or None
        If not None, the asymptotic level $g_*$ is set to this constant
        \textup{(}forward theoretical mode with a prescribed
        $g_*$\textup{)}. If None, the :py:meth:`forward` method returns
        $w_p(t)\,\mathrm{NN}(t)$, corresponding to $g_*=0$;
        the practical-calibrated mode with $g(y) = c_0 + \tau\,
        T^{-1}\!\int_0^T y$ is handled separately by
        :py:meth:`forward_with_g`.
    hidden : tuple of ints
        Architecture of the MLP.
    """

    def __init__(
        self,
        alpha: torch.Tensor,
        p: torch.Tensor,
        decay_exp: torch.Tensor,
        G_const: Optional[torch.Tensor] = None,
        hidden: Sequence[int] = (32, 32, 32, 32),
    ):
        super().__init__()
        self.alpha = alpha
        self.p = p
        # Frozen architectural exponent (no gradient)
        self.register_buffer("decay_exp", torch.as_tensor(
            decay_exp, dtype=torch.float64
        ))
        self.G_const = G_const
        self.mlp = MLP(hidden)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        decay = (1.0 + t).pow(self.decay_exp)
        N_theta = self.mlp(t)
        if self.G_const is None:
            return decay * N_theta
        return self.G_const + decay * N_theta

    def forward_with_g(
        self,
        t: torch.Tensor,
        c_0: torch.Tensor,
        tau: torch.Tensor,
        T_window: float,
        h: float,
        n_inner_iter: int = 2,
    ) -> torch.Tensor:
        r"""
        Compute $y_{NN}$ with $g(y) = c_0 + \tau T^{-1}\!\int_0^T y(s)\,ds$
        evaluated by trapezoidal rule on the mesh, via a short
        fixed-point iteration. Two passes are sufficient because the map
        $g(\cdot;c_0,\tau,T)$ is a weak contraction in $\tau \in [0,1)$.

        Uses the same frozen architectural decay exponent ``self.decay_exp``
        as :py:meth:`forward`, preserving the identifiability convention
        documented in the class docstring (the exponent is not a trainable
        parameter and does not depend on the current value of
        ``self.alpha``).
        """
        decay = (1.0 + t).pow(self.decay_exp)
        N_theta = self.mlp(t)
        # Initial guess: G_0 = c_0 (i.e. tau = 0)
        G_param = c_0
        # Find indices of the mesh in [0, T_window]
        mask = t <= T_window
        n_T = int(mask.sum().item())
        if n_T < 2:
            raise ValueError(
                f"T_window={T_window} too small for the mesh; need at least 2 nodes."
            )
        for _ in range(n_inner_iter):
            y = G_param + decay * N_theta
            # Trapezoidal average over [0, T_window]
            y_window = y[:n_T]
            avg = h / T_window * (
                0.5 * y_window[0] + y_window[1:-1].sum() + 0.5 * y_window[-1]
            )
            G_param = c_0 + tau * avg
        return G_param + decay * N_theta, G_param


# ---------------------------------------------------------------------------
# 3.  Loss functional
# ---------------------------------------------------------------------------


def fpinn_loss(
    y_NN: torch.Tensor,
    aF: torch.Tensor,
    h: float,
    alpha: torch.Tensor,
    *,
    y_obs: Optional[torch.Tensor] = None,
    obs_indices: Optional[torch.Tensor] = None,
    lambda_R: float = 1.0,
    lambda_D: float = 0.0,
    lambda_0: float = 0.0,
    y0_target: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    r"""
    Compute the fPINN loss with the residual restricted to the interior
    index set I_h = {0, 1, ..., floor(N/2)}:

        L = lambda_R * (1/|I_h|) * sum_{n in I_h}
                                | L_h y_NN(t_n) - aF(t_n) |^2
          + lambda_D * (1/m) * sum_j | y_NN(s_j) - y_obs(s_j) |^2
          + lambda_0 * | y_NN(0) - y0_target |^2

    Parameters
    ----------
    y_NN : (N+1,) tensor
        Mesh values of the network.
    aF : (N,) tensor
        Values of $a(t_n) F(y)(t_n)$ for $n = 0, ..., N-1$ (the right-hand
        side of the EDF that the residual must equal).
    h : float
        Mesh size.
    alpha : 0-d tensor
        Fractional order.
    y_obs, obs_indices : optional
        Observation data and their indices in the mesh; used iff lambda_D > 0.

    Returns
    -------
    loss : 0-d tensor
    breakdown : dict with separate terms
    """
    Lh = caputo_minus_L1_torch(y_NN, h, alpha)
    res = Lh - aF                                # (N,)
    # Restrict the residual loss to the interior index set
    # I_h = {0, ..., floor(N/2)}, on which Theorem 8.1 of the manuscript
    # provides uniform control of the L1 consistency error. The boundary
    # nodes n > floor(N/2) are excluded from the residual to align the
    # training functional with the rigorous variational consistency
    # theorem (Theorem 8.2).
    N_interior = len(res) // 2 + 1               # |I_h| = floor(N/2) + 1
    L_R = (res[:N_interior] ** 2).mean()
    breakdown = {"L_R": L_R.item()}

    L_total = lambda_R * L_R

    if lambda_D > 0 and y_obs is not None and obs_indices is not None:
        diff = y_NN[obs_indices] - y_obs
        L_D = (diff ** 2).mean()
        breakdown["L_D"] = L_D.item()
        L_total = L_total + lambda_D * L_D

    if lambda_0 > 0:
        L_0 = (y_NN[0] - y0_target) ** 2
        breakdown["L_0"] = L_0.item()
        L_total = L_total + lambda_0 * L_0

    breakdown["L_total"] = L_total.item()
    return L_total, breakdown


# ---------------------------------------------------------------------------
# 4.  Trainer (forward problem on y_ext)
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    n_iter: int = 5000
    lr: float = 1e-3
    log_every: int = 500
    tol: float = 1e-10


def train_forward_yext(
    alpha_val: float,
    gamma_val: float,
    h: float,
    T_max: float,
    config: Optional[TrainConfig] = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    r"""
    Train an asymptotically-aware fPINN to solve the forward problem with
    extremal-datum coefficients (a(s) = (1+s)^{-gamma}, f = 1, K = 0,
    g = 0). The exact solution is

        y_ext(t) = B(alpha, gamma - alpha) / Gamma(alpha) * (1+t)^{alpha - gamma}.

    With p = gamma in the architecture, the network must learn a constant
    N_theta(t) = B(alpha,gamma-alpha)/Gamma(alpha), which is a sanity check
    of the implementation.

    Returns a dict with keys:
        t (N+1,), y_NN (N+1,), y_exact (N+1,), L_history (n_iter,),
        final_err_inf (float).
    """
    if config is None:
        config = TrainConfig()
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = int(round(T_max / h))
    t_np = h * np.arange(N + 1)
    t = torch.from_numpy(t_np)

    # Right-hand side of the EDF: a(t_n) for n = 0, ..., N-1
    a_vals = (1.0 + t[:-1]).pow(-gamma_val)   # since f=1, F=1, K=0

    # Architecture: y_NN(t) = (1+t)^{alpha - gamma} * N_theta(t)  (G=0)
    alpha = torch.tensor(alpha_val, dtype=torch.float64)
    p = torch.tensor(gamma_val, dtype=torch.float64)
    decay_exp = float(alpha_val - gamma_val)
    net = AsymptoticallyAwareNet(
        alpha=alpha, p=p, decay_exp=decay_exp, G_const=None,
    )

    optimiser = torch.optim.Adam(net.parameters(), lr=config.lr)
    L_history: list[float] = []

    for it in range(config.n_iter):
        optimiser.zero_grad()
        y_NN = net(t)
        loss, _ = fpinn_loss(y_NN, a_vals, h, alpha, lambda_R=1.0, lambda_D=0.0)
        loss.backward()
        optimiser.step()
        L_history.append(loss.item())
        if verbose and (it == 0 or (it + 1) % config.log_every == 0):
            from scipy.special import beta as B_fn, gamma as G_fn
            with torch.no_grad():
                y_NN_eval = net(t).numpy()
                y_exact = (B_fn(alpha_val, gamma_val - alpha_val) / G_fn(alpha_val)) \
                          * (1.0 + t_np) ** (alpha_val - gamma_val)
                err = np.max(np.abs(y_NN_eval - y_exact))
            print(f"  iter {it+1:5d}: loss = {loss.item():.3e},  "
                  f"||y_NN - y_ext||_inf = {err:.3e}")
        if loss.item() < config.tol:
            if verbose:
                print(f"  converged at iter {it+1}")
            break

    with torch.no_grad():
        y_NN_final = net(t).numpy()
    from scipy.special import beta as B_fn, gamma as G_fn
    y_exact = (B_fn(alpha_val, gamma_val - alpha_val) / G_fn(alpha_val)) \
              * (1.0 + t_np) ** (alpha_val - gamma_val)
    err_inf = float(np.max(np.abs(y_NN_final - y_exact)))

    return dict(
        t=t_np, y_NN=y_NN_final, y_exact=y_exact,
        L_history=np.array(L_history), final_err_inf=err_inf,
        n_iter_used=len(L_history),
    )
