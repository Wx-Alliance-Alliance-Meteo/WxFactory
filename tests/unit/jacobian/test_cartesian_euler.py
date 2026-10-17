import torch

from tests.unit.cartesian_fixture import CartesianTestCase, cartesian_config, cartesian_simulation


class CartesianEulerJacobianTestCase(CartesianTestCase):
    def test_analytic_jacobian_and_partition(self):
        rhs = self.sim.rhs.full
        state = self.sim.Q.clone()
        state[1:4] = self.random_like(state[1:4])
        direction = self.random_like(state)
        direction[[0, 4]] *= 0.01
        epsilon = 1e-4
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
                    torch.testing.assert_close(derivative(direction).reshape_as(state), expected, atol=2e-6, rtol=2e-6)
                solution = partition.solve(direction, 0.1, column_batch_size=2)
                residual = solution - 0.05 * partition.implicit(solution).reshape_as(state) - direction
                torch.testing.assert_close(residual, torch.zeros_like(residual), atol=2e-11, rtol=0)
