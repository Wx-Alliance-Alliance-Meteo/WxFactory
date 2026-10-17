import torch

from tests.unit.cartesian_fixture import CartesianTestCase
from wx_factory.common.definitions import gravity
from wx_factory.common.matmul import apply_op


class CartesianMetric5DTestCase(CartesianTestCase):
    def test_metric_inverse_determinant_and_reference_scaling(self):
        metric = self.sim.initial_state.metric
        self.assertEqual(metric.scales, (250.0, 500.0, 1000 / 6))
        for fields in (metric.volume, *metric.faces):
            for i, scale in enumerate((250.0, 500.0, 1000 / 6)):
                torch.testing.assert_close(fields.G_cov[i + 1, i + 1], torch.full_like(fields.height, scale**2))
                torch.testing.assert_close(fields.G_contra[i + 1, i + 1], torch.full_like(fields.height, scale**-2))
            torch.testing.assert_close(fields.volume_jacobian, torch.full_like(fields.height, 250 * 500 * 1000 / 6))
            torch.testing.assert_close(fields.G_cov[0, 0], -2 * gravity * fields.height)
            torch.testing.assert_close(fields.G_contra[4, 4], 2 * gravity * fields.height)
            product = torch.einsum("ab...,bc...->ac...", fields.G_cov, fields.G_contra)
            identity = torch.eye(5, dtype=product.dtype).reshape((5, 5) + (1,) * fields.height.ndim)
            torch.testing.assert_close(product, identity.expand_as(product))
            determinant = torch.linalg.det(fields.G_cov.movedim((0, 1), (-2, -1)))
            torch.testing.assert_close(determinant, -(fields.volume_jacobian**2))
            torch.testing.assert_close(fields.geopotential, gravity * fields.height)
            self.assertTrue(bool((fields.dG_cov[2, 0, 0] == -2 * gravity * metric.scales[2]).all()))
            gram = torch.einsum("ai...,aj...->ij...", fields.physical_basis, fields.physical_basis)
            torch.testing.assert_close(gram, fields.G_cov[1:4, 1:4])

    def test_geopotential_dfr_gradient_at_actual_faces(self):
        rhs = self.sim.rhs.full
        for direction, fields in enumerate(rhs.metric.faces):
            gradient = apply_op(rhs.metric.volume.geopotential, rhs.derivatives[direction])
            gradient += apply_op(fields.geopotential[rhs.mid_faces[direction]], rhs.corrections[direction])
            expected = torch.full_like(gradient, gravity * rhs.metric.scales[2] if direction == 2 else 0)
            torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-8)

    def test_physical_velocity_round_trip(self):
        fields = self.sim.initial_state.metric.volume
        physical = self.random_like(self.sim.Q[1:4])
        covariant = fields.covariant_from_physical(physical)
        torch.testing.assert_close(fields.physical_velocity(covariant), physical)
        for i, scale in enumerate(self.sim.initial_state.metric.scales):
            torch.testing.assert_close(covariant[i], scale * physical[i])
