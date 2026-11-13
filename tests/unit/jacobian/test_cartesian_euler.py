import torch

from tests.unit.cartesian_fixture import CartesianTestCase, cartesian_config, cartesian_simulation
from wx_factory.jacobian.euler_vertical import EulerVerticalLinearization


class CartesianEulerJacobianTestCase(CartesianTestCase):
    def test_analytic_jacobian_and_partition(self):
        rhs = self.sim.rhs.full
        state = self.sim.Q.clone()
        state[1:4] = self.random_like(state[1:4])
        direction = self.random_like(state)
        direction[[0, 4]] *= 0.01
        epsilon = 1e-5
        torch.testing.assert_close(rhs(state), rhs.implicit(state) + rhs.explicit(state), atol=1e-8, rtol=1e-10)
        partition = rhs.linearize_partitions(state)
        explicit, implicit = partition.explicit, partition.implicit
        for function, linearization in (
            (rhs, rhs.linearize(state)),
            (rhs.explicit, explicit),
            (rhs.implicit, implicit),
        ):
            expected = (function(state + epsilon * direction) - function(state - epsilon * direction)) / (2 * epsilon)
            torch.testing.assert_close(linearization(direction).reshape_as(state), expected, atol=2e-4, rtol=2e-4)

    def test_periodic_gresho_jacobians_and_column_solve(self):
        config = cartesian_config(651)
        config.x1 = config.z1 = 1
        config.y1 = 0.1
        config.num_solpts = 2
        config.num_elements_horizontal = 3
        config.lateral_boundary = "periodic"
        config.vertical_boundary = "periodic"
        for elements in (1, 2, 3):
            with self.subTest(vertical_elements=elements):
                config.num_elements_vertical = elements
                sim = cartesian_simulation(config)
                rhs = sim.rhs.full
                state = sim.Q.clone()
                state[1:4] += 0.01 * self.random_like(state[1:4])
                direction = self.random_like(state)
                direction[[0, 4]] *= 0.01
                partition = rhs.linearize_partitions(state)
                epsilon = 1e-5
                for function, derivative in (
                    (rhs, rhs.linearize(state)),
                    (rhs.implicit, partition.implicit),
                    (rhs.explicit, partition.explicit),
                ):
                    expected = (function(state + epsilon * direction) - function(state - epsilon * direction)) / (
                        2 * epsilon
                    )
                    half_step = (
                        function(state + epsilon / 2 * direction) - function(state - epsilon / 2 * direction)
                    ) / epsilon
                    expected = (4 * half_step - expected) / 3
                    torch.testing.assert_close(derivative(direction).reshape_as(state), expected, atol=2e-6, rtol=2e-6)
                solution = partition.solve(direction, 0.1, column_batch_size=2)
                residual = solution - 0.05 * partition.implicit(solution).reshape_as(state) - direction
                torch.testing.assert_close(residual, torch.zeros_like(residual), atol=2e-11, rtol=0)

    def test_resting_stratified_column_has_no_growing_mode(self):
        config = cartesian_config(666)
        config.z1 = 10000
        config.num_elements_horizontal = 1
        sim = cartesian_simulation(config)
        linearization = EulerVerticalLinearization(sim.rhs.full, sim.Q)
        lower, diagonal, upper = (block[..., linearization.rows] for block in linearization.blocks)
        columns, elements, size, _ = diagonal.shape
        matrix = torch.zeros((columns, elements * size, elements * size), dtype=diagonal.dtype)
        for e in range(elements):
            rows = slice(e * size, (e + 1) * size)
            matrix[:, rows, rows] = diagonal[:, e]
            if e > 0:
                matrix[:, rows, (e - 1) * size : e * size] = lower[:, e]
            if e < elements - 1:
                matrix[:, rows, (e + 1) * size : (e + 2) * size] = upper[:, e]
        # Check the full column spectrum, including modes that alternate between nodes.
        self.assertLess(torch.linalg.eigvals(matrix).real.max().item(), 1e-10)
