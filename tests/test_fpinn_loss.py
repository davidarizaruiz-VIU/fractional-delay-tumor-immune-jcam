"""
test_fpinn_loss.py
==================

Unit tests for :func:`fpinn.fpinn_loss`, the residual loss minimised by
the asymptotically-aware fPINN of Section 8.2 of the manuscript.

These tests pin down the key correctness property that the paper's
variational consistency analysis (Theorem 8.2) relies on:

    The loss averages the squared residual over the *interior* index
    set I_h = {0, 1, ..., floor(N/2)}, NOT over the full mesh.

This is the conceptual coherence point between code and theorem.
Run with::

    PYTHONPATH=src/numerical_schemes python3 -m pytest tests/test_fpinn_loss.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/numerical_schemes/ is on the import path regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_NS_DIR = _REPO_ROOT / "src" / "numerical_schemes"
if str(_NS_DIR) not in sys.path:
    sys.path.insert(0, str(_NS_DIR))

import numpy as np
import pytest
import torch

from fpinn import fpinn_loss, caputo_minus_L1_torch  # noqa: E402

torch.set_default_dtype(torch.float64)


def _build_residual_targets(
    N: int, h: float, alpha_val: float,
    int_residual: float, ext_residual: float,
):
    """Construct (y_NN, aF) such that:

        L_h y_NN - aF = int_residual   on   I_h = {0, ..., floor(N/2)},
        L_h y_NN - aF = ext_residual   on   {floor(N/2)+1, ..., N-1}.

    Strategy: pick any y_NN (here y_NN = 0), compute L_h y_NN, and choose
    aF so that the difference equals the prescribed piecewise-constant
    residual vector.
    """
    alpha = torch.tensor(alpha_val, dtype=torch.float64)
    y_NN = torch.zeros(N + 1, dtype=torch.float64)
    Lh = caputo_minus_L1_torch(y_NN, h, alpha)
    target_res = torch.empty(N, dtype=torch.float64)
    N_interior = N // 2 + 1
    target_res[:N_interior] = int_residual
    target_res[N_interior:] = ext_residual
    aF = Lh - target_res
    return y_NN, aF, alpha, N_interior


def test_loss_uses_interior_residual_only_unit_int_only():
    """Residual = 1 on I_h and 0 outside ⇒ L_R = 1.0."""
    N, h = 20, 0.1
    y_NN, aF, alpha, _ = _build_residual_targets(
        N=N, h=h, alpha_val=1.5,
        int_residual=1.0, ext_residual=0.0,
    )
    loss, br = fpinn_loss(y_NN, aF, h, alpha, lambda_R=1.0, lambda_D=0.0)
    assert pytest.approx(br["L_R"], abs=1e-12) == 1.0
    assert pytest.approx(loss.item(), abs=1e-12) == 1.0


def test_loss_ignores_boundary_residual():
    """Residual = 1 on I_h and 100 outside ⇒ L_R must still be 1.0,
    not influenced by the boundary nodes. This is the key regression
    test demonstrating the I_h restriction."""
    N, h = 20, 0.1
    y_NN, aF, alpha, _ = _build_residual_targets(
        N=N, h=h, alpha_val=1.5,
        int_residual=1.0, ext_residual=100.0,
    )
    loss, br = fpinn_loss(y_NN, aF, h, alpha, lambda_R=1.0, lambda_D=0.0)
    # If the loss were full-mesh, L_R would be a weighted average of 1
    # (over |I_h| nodes) and 100^2 (over N - |I_h| nodes), so on the
    # order of 5e3. The interior restriction must give exactly 1.0.
    assert pytest.approx(br["L_R"], abs=1e-12) == 1.0
    assert pytest.approx(loss.item(), abs=1e-12) == 1.0


def test_loss_constant_interior_residual_squares_correctly():
    """L_R = (1/|I_h|) * sum_{n in I_h} r_n^2; for constant r_n = c on
    I_h, this is c^2."""
    N, h = 32, 0.05
    for c in (0.5, 2.0, -3.0):
        y_NN, aF, alpha, _ = _build_residual_targets(
            N=N, h=h, alpha_val=1.7,
            int_residual=c, ext_residual=0.0,
        )
        loss, br = fpinn_loss(y_NN, aF, h, alpha, lambda_R=1.0,
                              lambda_D=0.0)
        assert pytest.approx(br["L_R"], abs=1e-12) == c * c


def test_interior_size_matches_paper_definition():
    """For N points (mesh size N+1), |I_h| = floor(N/2) + 1."""
    for N in (4, 10, 11, 20, 100, 101):
        _, _, _, N_int = _build_residual_targets(
            N=N, h=0.1, alpha_val=1.5,
            int_residual=0.0, ext_residual=0.0,
        )
        assert N_int == N // 2 + 1


def test_lambda_R_scaling():
    """Doubling lambda_R should double the residual contribution to the
    total loss (with no data term)."""
    N, h = 16, 0.1
    y_NN, aF, alpha, _ = _build_residual_targets(
        N=N, h=h, alpha_val=1.4,
        int_residual=1.0, ext_residual=0.0,
    )
    loss_1, _ = fpinn_loss(y_NN, aF, h, alpha, lambda_R=1.0, lambda_D=0.0)
    loss_2, _ = fpinn_loss(y_NN, aF, h, alpha, lambda_R=2.0, lambda_D=0.0)
    assert pytest.approx(loss_2.item(), abs=1e-12) == 2.0 * loss_1.item()


if __name__ == "__main__":
    # Allow running as a standalone script.
    test_loss_uses_interior_residual_only_unit_int_only()
    test_loss_ignores_boundary_residual()
    test_loss_constant_interior_residual_squares_correctly()
    test_interior_size_matches_paper_definition()
    test_lambda_R_scaling()
    print("All fpinn_loss unit tests passed.")
