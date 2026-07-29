"""Directional-derivative check for the analytic vertical Jacobian J1 (PartRosExp2).

J1 is the exact derivative of the discrete vertically-stiff partition f1 = rhs.implicit, including
the delta-lambda variation of the Rusanov dissipation speed. This test compares the assembled
J1 . v (via block-tridiagonal blocks_matvec) against a central finite-difference directional
derivative of f1. Finite differences -- not complex step -- because the flux uses torch.abs for
|u3|, whose complex modulus does not carry the first-order perturbation (complex step would silently
reproduce the frozen-lambda operator).

The base state carries a nonzero, single-signed vertical velocity so the evaluation is away from the
non-smooth set (u3 = 0, argmax switches); a second test checks that the at-rest state (u3 = 0) still
assembles a finite operator under the sgn(0) = 0 convention.
"""

import os
import unittest

import torch
import torch.autograd.forward_ad as fwad
from wx_test import WxTestCase

from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.common.definitions import idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_u3
from wx_factory.device import differentiable_mode
from wx_factory.rhs.vertical_jacobian import (
    assemble_j1_blocks_analytic,
    blocks_matvec,
    col_to_state,
    forcing_jac_prepare,
    forcing_jvp,
    j2_flux_matvec,
    j2_prepare,
    state_to_col,
)
from wx_factory.simulation import Simulation


class VerticalJacobianTestCase(WxTestCase):
    def setUp(self) -> None:
        super().setUp()
        here = os.path.dirname(os.path.realpath(__file__))
        schema = load_default_schema()
        self.config = Configuration(readfile(os.path.join(here, "vertical_jacobian_config.ini")), schema)
        self.sim = Simulation(self.config)
        self.rhs = self.sim.rhs.full  # the RHS object (has .metric/.geom); .implicit is f1

    def _base_state(self, seed: int, w_speed: float) -> torch.Tensor:
        """Perturbed, physical state with a single-signed vertical velocity (away from u3 = 0)."""
        Q = self.sim.initial_state.Q.clone().to(torch.float64)
        gen = torch.Generator().manual_seed(seed)
        noise = torch.rand(Q.shape, generator=gen, dtype=torch.float64) - 0.5
        Q = Q * (1.0 + 0.02 * noise)  # few-percent perturbation keeps rho, rho_theta positive
        # Give every velocity a strictly-positive, single-signed value in [0.5, 1.5] * w_speed, so
        # sgn(u) is stable under the FD steps (away from the u = 0 non-smooth set), for u1, u2 and u3.
        for idx in (idx_rho_u1, idx_rho_u2, idx_rho_u3):
            vfac = 0.5 + torch.rand(Q[idx_rho].shape, generator=gen, dtype=torch.float64)
            Q[idx] = w_speed * Q[idx_rho] * vfac
        return Q

    def test_directional_derivative(self) -> None:
        """J1 . v vs a central finite-difference directional derivative of f1.

        The comparison includes the well-balanced rho_w row at the two rigid walls, where the
        reflected velocity makes the advective interface flux and its derivative exactly zero.
        """
        Q = self._base_state(seed=1234, w_speed=2.0)

        gen = torch.Generator().manual_seed(9)
        v = (torch.rand(Q.shape, generator=gen, dtype=torch.float64) - 0.5) * Q.abs()

        # Assembled J1 . v
        L, A, U = assemble_j1_blocks_analytic(self.rhs, Q)
        vc = state_to_col(self.rhs, v)
        Jv = col_to_state(self.rhs, blocks_matvec(self.rhs, L, A, U, vc), v)

        # Central finite-difference directional derivative of f1 = rhs.implicit
        eps = 1.0e-6
        ref = (self.rhs.implicit(Q + eps * v) - self.rhs.implicit(Q - eps * v)) / (2.0 * eps)

        num = torch.linalg.norm(Jv - ref).item()
        den = torch.linalg.norm(ref).item()
        rel = num / den
        self.assertLess(rel, 1.0e-7, f"assembled J1.v vs FD directional derivative: rel err {rel:.3e}")

    def test_finite_at_zero_vertical_velocity(self) -> None:
        # At-rest bubble: u3 = 0 at every trace (the non-smooth set). sgn(0) = 0 must keep J1 finite.
        Q = self.sim.initial_state.Q.clone().to(torch.float64)
        L, A, U = assemble_j1_blocks_analytic(self.rhs, Q)
        for blk in (L, A, U):
            self.assertTrue(torch.isfinite(blk).all(), "J1 blocks contain non-finite entries at u3 = 0")

    def test_rhs_partition_identity(self) -> None:
        """The directly assembled partitions must reproduce the production RHS."""
        Q = self._base_state(seed=4321, w_speed=2.0)
        full = self.rhs(Q)
        split = self.rhs.implicit(Q) + self.rhs.explicit(Q)
        error = torch.linalg.norm(split - full).item()
        scale = torch.linalg.norm(full).item()
        self.assertLess(error / scale, 1.0e-13, f"||f1 + f2 - f_full|| / ||f_full|| = {error / scale:.3e}")

    def test_j2_matches_explicit_directional_derivative(self) -> None:
        """The Jacobian used by PartRosExp2 must differentiate the actual f2 operator."""
        Q = self._base_state(seed=2468, w_speed=2.0)
        gen = torch.Generator().manual_seed(10)
        v = (torch.rand(Q.shape, generator=gen, dtype=torch.float64) - 0.5) * Q.abs()

        j2_base = j2_prepare(self.rhs, Q)
        forcing_base = forcing_jac_prepare(self.rhs, Q)
        analytic = j2_flux_matvec(self.rhs, Q, v, j2_base) + forcing_jvp(self.rhs, Q, v, forcing_base)

        eps = 1.0e-6
        reference = (self.rhs.explicit(Q + eps * v) - self.rhs.explicit(Q - eps * v)) / (2.0 * eps)

        row_errors = []
        for row in range(reference.shape[0]):
            numerator = torch.linalg.norm(analytic[row] - reference[row]).item()
            denominator = torch.linalg.norm(reference[row]).item()
            row_errors.append(numerator / denominator if denominator else numerator)

        error = torch.linalg.norm(analytic - reference).item()
        scale = torch.linalg.norm(reference).item()
        self.assertLess(
            error / scale,
            1.0e-7,
            f"analytic J2 does not differentiate f2: relative error {error / scale:.3e}; "
            f"per-row errors {row_errors}",
        )

    @unittest.skipUnless(
        differentiable_mode(),
        "needs WX_FACTORY_DIFFERENTIABLE=1, which must be set before torch tensors are created",
    )
    def test_jacobians_match_forward_mode_autodiff(self) -> None:
        """Compare the analytic Jacobian actions with forward-mode AD."""
        Q = self._base_state(seed=1234, w_speed=2.0)
        gen = torch.Generator().manual_seed(9)
        v = (torch.rand(Q.shape, generator=gen, dtype=torch.float64) - 0.5) * Q.abs()

        def tangent_of(func):
            with fwad.dual_level():
                return fwad.unpack_dual(func(fwad.make_dual(Q, v))).tangent.clone()

        L, A, U = assemble_j1_blocks_analytic(self.rhs, Q)
        j1v = col_to_state(self.rhs, blocks_matvec(self.rhs, L, A, U, state_to_col(self.rhs, v)), v)
        j2v = j2_flux_matvec(self.rhs, Q, v, j2_prepare(self.rhs, Q))
        j2v = j2v + forcing_jvp(self.rhs, Q, v, forcing_jac_prepare(self.rhs, Q))

        for name, analytic, reference in (
            ("J1", j1v, tangent_of(self.rhs.implicit)),
            ("J2", j2v, tangent_of(self.rhs.explicit)),
            ("J1 + J2", j1v + j2v, tangent_of(self.rhs)),
        ):
            error = torch.linalg.norm(analytic - reference).item()
            scale = torch.linalg.norm(reference).item()
            self.assertLess(
                error / scale,
                1.0e-13,
                f"analytic {name} . v vs forward-mode AD: relative error {error / scale:.3e}",
            )

    def test_jacobian_partition_identity(self) -> None:
        """The two Jacobian actions used by PartRosExp2 must add to the full RHS Jacobian."""
        Q = self._base_state(seed=1357, w_speed=2.0)
        gen = torch.Generator().manual_seed(11)
        v = (torch.rand(Q.shape, generator=gen, dtype=torch.float64) - 0.5) * Q.abs()

        L, A, U = assemble_j1_blocks_analytic(self.rhs, Q)
        j1v = col_to_state(
            self.rhs,
            blocks_matvec(self.rhs, L, A, U, state_to_col(self.rhs, v)),
            v,
        )
        j2v = j2_flux_matvec(self.rhs, Q, v, j2_prepare(self.rhs, Q))
        j2v += forcing_jvp(self.rhs, Q, v, forcing_jac_prepare(self.rhs, Q))

        eps = 1.0e-6
        reference = (self.rhs(Q + eps * v) - self.rhs(Q - eps * v)) / (2.0 * eps)
        error = torch.linalg.norm(j1v + j2v - reference).item()
        scale = torch.linalg.norm(reference).item()
        self.assertLess(
            error / scale,
            1.0e-7,
            f"||J1.v + J2.v - J_full.v|| / ||J_full.v|| = {error / scale:.3e}",
        )
