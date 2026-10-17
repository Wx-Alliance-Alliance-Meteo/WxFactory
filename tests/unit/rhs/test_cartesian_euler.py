import torch

from tests.unit.cartesian_fixture import CartesianTestCase, cartesian_config, cartesian_simulation
from wx_factory.common.definitions import gravity
from wx_factory.pde.euler_interfaces import acoustic_interface


class CartesianEulerTestCase(CartesianTestCase):
    def test_walls_and_closed_domain_scalar_conservation(self):
        rhs = self.sim.rhs.full
        state = self.sim.Q.clone()
        state[1:4] = 0.1 * self.random_like(state[1:4])
        faces = rhs._faces_with_neighbours(rhs._extrapolate(state))
        width = self.sim.geometry.itf_size
        for direction, face in enumerate(faces):
            boundary_faces = face.movedim(3 - direction, 1)
            for ghost, interior, ghost_half, interior_half in (
                (0, 1, slice(width, None), slice(None, width)),
                (-1, -2, slice(None, width), slice(width, None)),
            ):
                expected = boundary_faces[:, interior, ..., interior_half].clone()
                expected[direction + 1] *= -1
                torch.testing.assert_close(boundary_faces[:, ghost, ..., ghost_half], expected, atol=1e-12, rtol=1e-12)
        for direction, (geometry, face) in enumerate(zip(rhs._static_interfaces(), faces)):
            normal, _, _ = acoustic_interface(
                face[geometry.left],
                face[geometry.right],
                geometry.inverse_spatial,
                direction,
                geometry.geopotential,
                wall=geometry.wall,
            )
            self.assertTrue(bool((normal[geometry.wall] == 0).all()))
        value = rhs(state)
        weights = self.sim.geometry.glweights
        volume_weights = torch.einsum("k,j,i->kji", weights, weights, weights).flatten()
        total = (value[[0, 4]] * rhs.metric.volume.volume_jacobian * volume_weights).sum(dim=(1, 2, 3, 4))
        torch.testing.assert_close(total, torch.zeros_like(total), atol=1e-6, rtol=0)

    def test_periodic_horizontal_translation_and_gravity(self):
        config = cartesian_config()
        config.lateral_boundary = "periodic"
        sim = cartesian_simulation(config)
        state = torch.zeros_like(sim.Q)
        state[0], state[4] = 1, 300
        state[1] = 2 * sim.initial_state.metric.scales[0]
        expected = torch.zeros_like(state)
        expected[3] = -gravity * sim.initial_state.metric.scales[2]
        torch.testing.assert_close(sim.rhs.full(state), expected, rtol=1e-12, atol=2e-8)
        rhs = sim.rhs.full
        state[0] = 1 + 0.1 * torch.sin(2 * torch.pi * sim.geometry.X1 / 1000)
        profile = sim.geometry.to_single_block(state)[0, 0, 0].reshape(2, config.num_solpts)
        right = torch.exp(torch.log(profile) @ rhs.ops.extrap_east)
        common = torch.stack((right.roll(1, dims=0), right), dim=-1)
        expected = -2 / rhs.metric.scales[0] * (profile @ rhs.ops.diff_solpt_tr + common @ rhs.ops.correction_tr)
        actual = sim.geometry.to_single_block(rhs(state))[0]
        torch.testing.assert_close(actual, expected.flatten().expand_as(actual), atol=1e-12, rtol=1e-10)

    def test_periodic_gresho_rotation_conservation_and_vertical_transport(self):
        config = cartesian_config(651)
        config.x1 = config.z1 = 1
        config.y1 = 0.1
        config.num_solpts = 2
        config.num_elements_horizontal = config.num_elements_vertical = 3
        config.lateral_boundary = config.vertical_boundary = "periodic"
        sim = cartesian_simulation(config)
        rhs = sim.rhs.full
        state = sim.Q.clone()
        state[1:4] += 0.01 * self.random_like(state[1:4])

        def rotate(value):
            block = torch.rot90(sim.geometry.to_single_block(value), 1, dims=(-3, -1))
            rotated = block.clone()
            rotated[1], rotated[3] = block[3], -block[1]
            return sim.geometry._to_new(rotated)

        value = rhs(state)
        torch.testing.assert_close(rhs(rotate(state)), rotate(value), atol=2e-10, rtol=2e-10)
        weights = sim.geometry.glweights
        weights = torch.einsum("k,j,i->kji", weights, weights, weights).flatten()
        total = (value[[0, 4]] * rhs.metric.volume.volume_jacobian * weights).sum(dim=(1, 2, 3, 4))
        torch.testing.assert_close(total, torch.zeros_like(total), atol=1e-11, rtol=0)

        # A vertical contact crosses the periodic boundary even with lateral walls.
        for lateral in ("periodic", "wall"):
            with self.subTest(lateral=lateral):
                config.lateral_boundary = lateral
                sim = cartesian_simulation(config)
                rhs = sim.rhs.full
                state = torch.zeros_like(sim.Q)
                state[0] = 1 + 0.1 * torch.sin(2 * torch.pi * sim.geometry.X3)
                state[4] = 300
                state[3] = 2 * rhs.metric.scales[2]
                profile = sim.geometry.to_single_block(state)[0, :, 0, 0].reshape(3, config.num_solpts)
                right = torch.exp(torch.log(profile) @ rhs.ops.extrap_up)
                common = torch.stack((right.roll(1, dims=0), right), dim=-1)
                expected = (
                    -2 / rhs.metric.scales[2] * (profile @ rhs.ops.diff_solpt_tr + common @ rhs.ops.correction_tr)
                )
                actual = sim.geometry.to_single_block(rhs(state))[0]
                torch.testing.assert_close(
                    actual, expected.flatten()[:, None, None].expand_as(actual), atol=1e-11, rtol=1e-10
                )
