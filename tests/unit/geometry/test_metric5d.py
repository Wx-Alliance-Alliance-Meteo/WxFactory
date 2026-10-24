import unittest

import torch

from wx_factory.geometry.metric5d import evaluate_metric5d, panel_frame
from wx_factory.geometry.terrain_mapping import angular_distance_squared, cosine_of_distance


class Metric5DTestCase(unittest.TestCase):
    def setUp(self):
        self.points = torch.tensor([[0.12, -0.31], [-0.24, 0.27], [0.2, 0.7]], dtype=torch.float64)
        self.frame = panel_frame(0.4, 0.3, -0.2, self.points)
        self.scales = (0.07, 0.13, 0.2)

    @staticmethod
    def terrain(points):
        x, y, eta = points
        return 3 * eta + (1 - eta) ** 2 * (0.2 + 0.1 * torch.sin(2 * x) * torch.cos(y))

    def metric(self, points=None, scales=None, deep=True, rotation=0.08):
        return evaluate_metric5d(
            self.points if points is None else points,
            self.terrain,
            radius=7.0,
            rotation_speed=rotation,
            frame=self.frame,
            scales=self.scales if scales is None else scales,
            deep=deep,
        )

    def test_chunked_metric_preserves_all_fields_and_shape(self):
        from dataclasses import fields

        from wx_factory.geometry.metric5d import Metric5D, Metric5DFields

        points = self.points.repeat(1, 5).reshape(3, 2, 5)
        metric = Metric5D.__new__(Metric5D)
        batch_sizes = []

        def evaluate(batch):
            batch_sizes.append(batch.shape[-1])
            return self.metric(batch)

        metric.evaluate = evaluate
        expected = self.metric(points)
        for cpu_storage in (False, True):
            batch_sizes.clear()
            actual = metric._evaluate_in_chunks(points, cpu_storage=cpu_storage, chunk_size=4)
            self.assertLessEqual(max(batch_sizes), 4)
            for field in fields(Metric5DFields):
                with self.subTest(field=field.name, cpu_storage=cpu_storage):
                    torch.testing.assert_close(
                        getattr(actual, field.name), getattr(expected, field.name), rtol=0, atol=0
                    )
            for name in ("frame_vorticity", "vertical_frame_gradient", "horizontal_frame_gradient"):
                torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)

    def test_full_inverse_determinant_and_spatial_basis(self):
        for deep in (False, True):
            with self.subTest(deep=deep):
                fields = self.metric(deep=deep)
                product = torch.einsum("ab...,bc...->ac...", fields.G_cov, fields.G_contra)
                torch.testing.assert_close(
                    product, torch.eye(5, dtype=torch.float64)[:, :, None].expand_as(product), atol=2e-13, rtol=2e-13
                )
                determinant = torch.linalg.det(fields.G_cov.movedim((0, 1), (-2, -1)))
                torch.testing.assert_close(determinant, -(fields.volume_jacobian**2))
                gram = torch.einsum("ai...,aj...->ij...", fields.physical_basis, fields.physical_basis)
                torch.testing.assert_close(gram, fields.G_cov[1:4, 1:4], atol=1e-14, rtol=1e-14)
                self.assertTrue(
                    bool((torch.linalg.eigvalsh(fields.G_cov.movedim((0, 1), (-2, -1))) < 0).sum(dim=-1).eq(1).all())
                )

    def test_inverse_and_wind_conversion_with_small_vertical_jacobian(self):
        for deep in (False, True):
            for vertical_slope in (3.0, 2e-4):
                with self.subTest(deep=deep, vertical_slope=vertical_slope):

                    def terrain(points, vertical_slope=vertical_slope):
                        return 0.3 + 0.2 * torch.sin(points[0]) + 0.1 * points[1] + vertical_slope * points[2]

                    fields = evaluate_metric5d(
                        self.points,
                        terrain,
                        radius=7.0,
                        rotation_speed=0.08,
                        frame=self.frame,
                        scales=self.scales,
                        deep=deep,
                    )
                    spatial = fields.G_cov[1:4, 1:4]
                    inverse = fields.G_contra[1:4, 1:4]
                    reference = torch.linalg.inv(spatial.movedim((0, 1), (-2, -1))).movedim((-2, -1), (0, 1))
                    torch.testing.assert_close(inverse, reference, rtol=5e-10, atol=1e-12)
                    product = torch.einsum("ij...,jk...->ik...", spatial, inverse)
                    torch.testing.assert_close(
                        product, torch.eye(3, dtype=torch.float64)[:, :, None].expand_as(product), rtol=0, atol=5e-10
                    )
                    physical = torch.tensor([[0.2, -0.1], [0.4, 0.3], [-0.2, 0.1]], dtype=torch.float64)
                    recovered = fields.physical_velocity(fields.covariant_from_physical(physical))
                    torch.testing.assert_close(recovered, physical, rtol=5e-10, atol=1e-12)

    def test_all_components_against_independent_embedded_line_element(self):
        # Differentiate the Cartesian embedding to obtain an independent metric.
        from wx_factory.common.definitions import gravity

        with torch.inference_mode(False), torch.enable_grad():
            for deep in (False, True):
                actual = self.metric(deep=deep)
                for point_index in range(self.points.shape[1]):
                    base = self.points[:, point_index]
                    scales = base.new_tensor(self.scales)

                    def embedding(reference, base=base, scales=scales, deep=deep):
                        point = base + scales * reference
                        raw = torch.stack((torch.tan(point[0]), torch.tan(point[1]), point.new_tensor(1.0)))
                        radial = self.frame @ (raw / torch.linalg.vector_norm(raw))
                        height = self.terrain(point[:, None])[0]
                        if deep:
                            return (7.0 + height) * radial
                        # Shallow spatial line element is the product of the
                        # reference sphere and the independent height coordinate.
                        return torch.cat((7.0 * radial, height[None]))

                    reference = torch.zeros(3, dtype=torch.float64)
                    position = embedding(reference)
                    basis = torch.autograd.functional.jacobian(embedding, reference)
                    height = self.terrain(base[:, None])[0]
                    potential = gravity * 7.0 * height / (7.0 + height) if deep else gravity * height
                    expected = torch.zeros((5, 5), dtype=torch.float64)
                    expected[0, 0] = -2 * potential
                    expected[0, 4] = expected[4, 0] = 1.0
                    expected[1:4, 1:4] = basis.T @ basis
                    omega = base.new_tensor([0.0, 0.0, 0.08])
                    if deep:
                        potential -= 0.5 * torch.linalg.cross(omega, position).square().sum()
                        expected[0, 0] = -2 * potential
                    shift = torch.linalg.cross(omega, position[:3]) @ basis[:3]
                    expected[0, 1:4] = expected[1:4, 0] = shift
                    torch.testing.assert_close(actual.G_cov[:, :, point_index], expected, rtol=3e-13, atol=3e-13)
                    torch.testing.assert_close(
                        actual.G_contra[:, :, point_index], torch.linalg.inv(expected), rtol=3e-13, atol=3e-13
                    )
                    torch.testing.assert_close(
                        actual.volume_jacobian[point_index],
                        torch.sqrt(torch.linalg.det(basis.T @ basis)),
                        rtol=3e-13,
                        atol=3e-13,
                    )

    def test_static_derivative_offload_preserves_full_tensor_and_frame_curl(self):
        from wx_factory.geometry.metric5d import Metric5D

        for deep in (False, True):
            metric = Metric5D.__new__(Metric5D)
            metric.volume = self.metric(deep=deep)
            metric.faces = ()
            expected_derivatives = metric.volume.dG_cov.clone()
            expected_curl = metric.volume.frame_vorticity.clone()
            metric.offload_derivatives()
            torch.testing.assert_close(metric.volume.dG_cov, expected_derivatives, rtol=0, atol=0)
            torch.testing.assert_close(metric.volume.frame_vorticity, expected_curl, rtol=0, atol=0)
            metric.to_runtime(torch.float32)
            derivatives = expected_derivatives.to(torch.float32)[:, 0, 1:4]
            torch.testing.assert_close(
                metric.volume.frame_vorticity, derivatives - derivatives.transpose(0, 1), rtol=0, atol=0
            )

    def test_reference_element_pullback(self):
        panel = self.metric(scales=(1.0, 1.0, 1.0))
        reference = self.metric()
        scale = self.points.new_tensor((1.0, *self.scales, 1.0))
        torch.testing.assert_close(reference.G_cov, panel.G_cov * scale[:, None, None] * scale[None, :, None])
        torch.testing.assert_close(reference.G_contra, panel.G_contra / scale[:, None, None] / scale[None, :, None])
        torch.testing.assert_close(
            reference.volume_jacobian, panel.volume_jacobian * self.scales[0] * self.scales[1] * self.scales[2]
        )
        for k in range(3):
            torch.testing.assert_close(
                reference.dG_cov[k], panel.dG_cov[k] * self.scales[k] * scale[:, None, None] * scale[None, :, None]
            )

    def test_first_derivatives_against_centered_differences(self):
        for deep in (False, True):
            fields = self.metric(deep=deep)
            for direction in range(3):
                perturbation = torch.zeros_like(self.points)
                step = 2e-5
                perturbation[direction] = step * self.scales[direction]
                difference = (
                    self.metric(self.points + perturbation, deep=deep).G_cov
                    - self.metric(self.points - perturbation, deep=deep).G_cov
                ) / (2 * step)
                torch.testing.assert_close(fields.dG_cov[direction], difference, atol=2e-9, rtol=2e-8)

    def test_deep_effective_gravity_against_cartesian_force(self):
        from wx_factory.common.definitions import gravity

        fields = self.metric()
        radius = 7.0 + fields.height
        position = radius * fields.radial
        omega = position.new_tensor([0.0, 0.0, 0.08])[:, None].expand_as(position)
        acceleration = -gravity * (7.0 / radius) ** 2 * fields.radial - torch.linalg.cross(
            omega, torch.linalg.cross(omega, position, dim=0), dim=0
        )
        expected_gradient = -torch.einsum("ai...,a...->i...", fields.physical_basis, acceleration)
        torch.testing.assert_close(-0.5 * fields.dG_cov[:, 0, 0], expected_gradient, atol=2e-14, rtol=2e-14)

    def test_deep_effective_potential_from_metric_shift(self):
        from wx_factory.common.definitions import gravity

        for rotation in (0.0, 0.08):
            fields = self.metric(rotation=rotation)
            shift = fields.G_cov[0, 1:4]
            frame_energy = 0.5 * torch.einsum("i...,ij...,j...->...", shift, fields.G_contra[1:4, 1:4], shift)
            gravitational_potential = gravity * 7.0 * fields.height / (7.0 + fields.height)
            torch.testing.assert_close(fields.geopotential, gravitational_potential - frame_energy)
            torch.testing.assert_close(fields.G_contra[4, 4], 2 * gravitational_potential)

    def test_wind_roundtrip_energy_and_wall_reflection(self):
        velocity = self.points.new_tensor([[1.2, -2.3], [0.3, 0.4], [-0.7, 1.8]])
        for deep in (False, True):
            fields = self.metric(deep=deep)
            covariant = fields.covariant_from_physical(velocity)
            torch.testing.assert_close(fields.physical_velocity(covariant), velocity)
            kinetic = (fields.raise_velocity(covariant) * covariant).sum(dim=0)
            torch.testing.assert_close(kinetic, (velocity * velocity).sum(dim=0))
            for direction in range(3):
                reflected = fields.reflect(covariant, direction)
                torch.testing.assert_close(fields.reflect(reflected, direction), covariant)
                torch.testing.assert_close(
                    fields.raise_velocity(reflected)[direction], -fields.raise_velocity(covariant)[direction]
                )
                torch.testing.assert_close((fields.physical_velocity(reflected) ** 2).sum(dim=0), kinetic)

    def test_frame_vorticity_sign_no_work_and_shallow_limit(self):
        wind = self.points.new_tensor([[1.2, -2.3], [0.3, 0.4], [-0.7, 1.8]])
        omega = torch.zeros_like(wind)
        omega[2] = 0.08
        for deep in (False, True):
            fields = self.metric(deep=deep)
            covariant = fields.covariant_from_physical(wind)
            velocity = fields.raise_velocity(covariant)
            force = torch.einsum("ij...,j...->i...", fields.frame_vorticity, velocity)
            effective_omega = omega if deep else (omega * fields.radial).sum(dim=0) * fields.radial
            expected = -2 * torch.linalg.cross(effective_omega, wind, dim=0)
            torch.testing.assert_close(fields.physical_velocity(force), expected, atol=1e-14, rtol=1e-13)
            torch.testing.assert_close(
                (velocity * force).sum(dim=0), torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0.0
            )
            if not deep:
                torch.testing.assert_close(fields.frame_vorticity[2], torch.zeros_like(fields.frame_vorticity[2]))
        torch.testing.assert_close(
            self.metric(rotation=0.0).frame_vorticity, torch.zeros((3, 3, 2), dtype=torch.float64)
        )

    def test_affine_height_has_zero_hessian(self):
        fields = evaluate_metric5d(
            self.points,
            lambda x: 3 * x[2],
            radius=7.0,
            rotation_speed=0.0,
            frame=self.frame,
            scales=self.scales,
            deep=False,
        )
        torch.testing.assert_close(fields.dG_cov[:, 3, 3], torch.zeros((3, 2), dtype=torch.float64))

    def test_mountain_centre_limits(self):
        # The radial functions are smooth at the centre despite the singular acos derivative.
        with torch.inference_mode(False), torch.enable_grad():
            cosine = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
            for function, value, first, second in (
                (angular_distance_squared, 0.0, -2.0, 2 / 3),
                (lambda c: cosine_of_distance(c, 32.0), 1.0, 32**2, 32**2 * (32**2 - 1) / 3),
            ):
                result = function(cosine)
                derivative = torch.autograd.grad(result, cosine, create_graph=True)[0]
                curvature = torch.autograd.grad(derivative, cosine)[0]
                self.assertAlmostEqual(result.item(), value)
                self.assertAlmostEqual(derivative.item(), first)
                self.assertAlmostEqual(curvature.item(), second)

    def test_mountain_ripple_resolves_high_frequency_near_centre(self):
        with torch.inference_mode(False), torch.enable_grad():
            for frequency in (32.0, 10008.0):
                phase = torch.tensor([0.0, 1e-5, 1e-3, 0.004, 0.005, 0.01, 0.1, 1.0, 4.0], dtype=torch.float64)
                angle = (phase / frequency).requires_grad_()
                result = cosine_of_distance(torch.cos(angle), frequency)
                derivative = torch.autograd.grad(result.sum(), angle, create_graph=True)[0]
                curvature = torch.autograd.grad(derivative.sum(), angle)[0]
                with self.subTest(frequency=frequency):
                    torch.testing.assert_close(result, torch.cos(phase), rtol=2e-8, atol=2e-8)
                    torch.testing.assert_close(derivative / frequency, -torch.sin(phase), rtol=2e-8, atol=2e-8)
                    torch.testing.assert_close(curvature / frequency**2, -torch.cos(phase), rtol=2e-8, atol=2e-8)

    def test_rejects_folded_vertical_map(self):
        with self.assertRaisesRegex(ValueError, "orientation-preserving"):
            evaluate_metric5d(
                self.points,
                lambda x: -x[2],
                radius=7.0,
                rotation_speed=0.0,
                frame=self.frame,
                scales=self.scales,
                deep=False,
            )


class TabulatedSurfaceTestCase(unittest.TestCase):
    def test_polynomial_height_gradient_and_hessian(self):
        from wx_factory.geometry.metric5d import mapping_derivatives
        from wx_factory.geometry.terrain_mapping import TabulatedSurface

        x = torch.tensor([-0.7, -0.2, 0.1, 0.4, 0.8], dtype=torch.float64)
        y = torch.tensor([-0.6, -0.3, 0.15, 0.5, 0.9], dtype=torch.float64)
        surface = 200 + 30 * y[:, None] ** 2 + 40 * x[None] ** 3 + 20 * y[:, None] * x[None]
        interpolant = TabulatedSurface(x, y, surface)
        points = torch.tensor([[-0.4, 0.0, 0.55], [-0.1, 0.4, 0.7], [0.2, 0.5, 0.8]], dtype=torch.float64)
        height, gradient, hessian = mapping_derivatives(interpolant, points)
        xx, yy, _ = points
        torch.testing.assert_close(height, 200 + 30 * yy**2 + 40 * xx**3 + 20 * yy * xx)
        torch.testing.assert_close(gradient[0], 120 * xx**2 + 20 * yy)
        torch.testing.assert_close(gradient[1], 60 * yy + 20 * xx)
        torch.testing.assert_close(gradient[2], torch.zeros_like(xx), atol=1e-11, rtol=0)
        torch.testing.assert_close(hessian[0, 0], 240 * xx, atol=1e-10, rtol=1e-12)
        torch.testing.assert_close(hessian[0, 1], torch.full_like(xx, 20), atol=1e-10, rtol=1e-12)
        torch.testing.assert_close(hessian[1, 1], torch.full_like(xx, 60), atol=1e-10, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
