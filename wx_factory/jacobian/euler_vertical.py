"""Column blocks and complementary Jacobian action for the Euler partitions."""

import torch

from ..common.definitions import heat_capacity_ratio
from ..common.matmul import apply_op
from ..pde.euler_interfaces import EulerInterfaceGeometry, pressure
from .block_solve import block_thomas_solve, blocks_matvec, cyclic_block_solve
from .column_layout import (
    column_dims,
    columns_to_state,
    scalar_interface_to_columns,
    scalar_to_columns,
    state_interface_to_columns,
    state_to_columns,
    stiff_variable_rows,
)
from .euler import AcousticLinearization


class EulerVerticalLinearization:
    """Assemble the (rho, v_3, rho*theta) rows against all five variables.

    Column states have axes (column, element, variable, node). Before flattening,
    each block has axes (column, element, equation, output node, variable, input node).
    The three equation rows are rho, v_3, and rho*theta.
    """

    def __init__(self, rhs, state, *, columns=None):
        self.rhs = rhs
        self.shape = state.shape
        state = state.reshape((-1,) + rhs.metric.volume.height.shape)
        self.state = state
        dimensions = column_dims(rhs, state)
        self.dimensions = dimensions
        self.column_slice = slice(None) if columns is None else columns
        self.rows = stiff_variable_rows(dimensions.num_solpts, state.device)
        self.state_columns = state_to_columns(rhs, state).reshape(
            dimensions.num_columns, dimensions.num_elem_z, 5, dimensions.num_solpts
        )[self.column_slice]
        if columns is not None:
            self.state_columns = self.state_columns.clone()
        self.derivative = rhs.ops.diff_solpt.to(state.dtype)
        self.correction = rhs.ops.correction.to(state.dtype)
        self.extrapolation = torch.stack((rhs.ops.extrap_down, rhs.ops.extrap_up)).to(state.dtype)

        def volume_columns(value):
            result = scalar_to_columns(value, dimensions)[self.column_slice]
            return result if columns is None else result.clone()

        def face_columns(value):
            result = scalar_interface_to_columns(value, dimensions)[self.column_slice]
            return result if columns is None else result.clone()

        fields = rhs.metric.volume
        self.jacobian = volume_columns(fields.volume_jacobian)
        self.inverse_spatial = torch.stack(
            [torch.stack([volume_columns(fields.G_contra[i + 1, j + 1]) for j in range(3)]) for i in range(3)]
        )
        self.contravariant_velocity = torch.einsum(
            "ijces,cejs->ceis", self.inverse_spatial, self.state_columns[:, :, 1:4]
        )

        state_trace = apply_op(state, rhs.extrapolations[2])
        state_trace[[0, 4]] = torch.exp(apply_op(torch.log(state[[0, 4]]), rhs.extrapolations[2]))
        self.state_trace = state_interface_to_columns(state_trace, dimensions)[self.column_slice].clone()
        faces = state_interface_to_columns(rhs._vertical_faces(state_trace), dimensions)[self.column_slice].clone()
        left = faces[:, :-1, 1].permute(2, 0, 1)
        right = faces[:, 1:, 0].permute(2, 0, 1)
        face_fields = rhs.metric.faces[2]
        inverse_faces = torch.stack(
            [torch.stack([face_columns(face_fields.G_contra[i + 1, j + 1]) for j in range(3)]) for i in range(3)]
        )
        inverse_spatial = inverse_faces[:, :, :, :-1, 1]
        geopotential = face_columns(face_fields.geopotential)[:, :-1, 1]
        jacobian = face_columns(face_fields.volume_jacobian)[:, :-1, 1]
        normal_scale = torch.sqrt(inverse_spatial[2, 2])
        normal_row = inverse_spatial[2] / normal_scale
        wall = None
        if not rhs.periodic_vertical:
            wall = torch.zeros_like(normal_scale, dtype=torch.bool)
            wall[:, 0], wall[:, -1] = True, True
        geometry = EulerInterfaceGeometry(
            left=None,
            right=None,
            inverse_spatial=inverse_spatial,
            geopotential=geopotential,
            jacobian=jacobian,
            normal_scale=normal_scale,
            normal_row=normal_row,
            wall=wall,
        )
        interface = AcousticLinearization(
            left,
            right,
            geometry,
            2,
            mach_cutoff=rhs.config.riemann_mach_cutoff,
            low_mach=rhs.config.riemann_solver == "low_mach",
        )
        self.pressure_gradient = self._vertical_derivative(
            pressure(self.state_columns[:, :, 4]), interface.common_pressure
        )
        self.vertical_covariant_gradient = torch.stack(
            [
                self._vertical_derivative(self.state_columns[:, :, j + 1], interface.common_covariant_velocity[j])
                for j in range(2)
            ],
            dim=2,
        )
        self.vertical_frame_gradient = torch.stack(
            [volume_columns(fields.vertical_frame_gradient[a]) for a in range(2)], dim=2
        )
        left_matrix, right_matrix = self._interface_jacobians(interface, left, jacobian)

        bottom_matrix = top_matrix = None
        if not rhs.periodic_vertical:
            # The reflected ghost's v_3 depends on all three interior covariant velocities.
            bottom = self._wall_reflection_matrix(inverse_faces[:, :, :, 0, 1])
            top = self._wall_reflection_matrix(inverse_faces[:, :, :, -1, 0])
            bottom_matrix = left_matrix[:, 0] @ bottom + right_matrix[:, 0]
            top_matrix = left_matrix[:, -1] + right_matrix[:, -1] @ top
        self.blocks = self._assemble_blocks(left_matrix, right_matrix, bottom_matrix, top_matrix)

    def _vertical_derivative(self, value, common):
        return (
            torch.einsum("os,ces->ceo", self.derivative, value)
            + self.correction[:, 0] * common[:, :-1, None]
            + self.correction[:, 1] * common[:, 1:, None]
        )

    def _wall_reflection_matrix(self, inverse_spatial):
        result = (
            torch.eye(5, dtype=self.state.dtype, device=self.state.device)
            .expand(self.state_columns.shape[0], 5, 5)
            .clone()
        )
        result[:, 3, 1:4] -= (2 * inverse_spatial[2] / inverse_spatial[2, 2]).T
        return result

    def _interface_jacobians(self, interface, left, jacobian):
        """Differentiate (J flux_rho, J flux_theta, p*, K*, v_1*, v_2*) on each side."""
        matrices = []
        zero_perturbation = torch.zeros_like(left)
        for side in range(2):
            columns = []
            for variable in range(5):
                delta = torch.zeros_like(left)
                delta[variable] = 1
                _, delta_pressure, delta_kinetic_energy, delta_covariant_velocity, flux = (
                    interface(delta, zero_perturbation) if side == 0 else interface(zero_perturbation, delta)
                )
                values = torch.cat(
                    (jacobian * flux, delta_pressure[None], delta_kinetic_energy[None], delta_covariant_velocity[:2]),
                    dim=0,
                )
                columns.append(values.permute(1, 2, 0))
            matrices.append(torch.stack(columns, dim=-1))
        return matrices

    def _assemble_blocks(self, left, right, bottom, top):
        lower, diagonal, upper = self._volume_blocks()
        extrapolation = self._extrapolation_jacobians()
        lifts = self._face_lifting_matrices()
        dimensions, state_columns = self.dimensions, self.state_columns

        def contribution(lift, interface, extrapolation):
            return torch.einsum("ceioq,ceqj,cejs->ceiojs", lift, interface, extrapolation)

        lower[:, 1:] += contribution(lifts[0][:, 1:], left[:, 1:-1], extrapolation[1][:, :-1])
        diagonal[:, 1:] += contribution(lifts[0][:, 1:], right[:, 1:-1], extrapolation[0][:, 1:])
        diagonal[:, :-1] += contribution(lifts[1][:, :-1], left[:, 1:-1], extrapolation[1][:, :-1])
        upper[:, :-1] += contribution(lifts[1][:, :-1], right[:, 1:-1], extrapolation[0][:, 1:])
        if self.rhs.periodic_vertical:
            lower[:, :1] += contribution(lifts[0][:, :1], left[:, :1], extrapolation[1][:, -1:])
            diagonal[:, :1] += contribution(lifts[0][:, :1], right[:, :1], extrapolation[0][:, :1])
            diagonal[:, -1:] += contribution(lifts[1][:, -1:], left[:, -1:], extrapolation[1][:, -1:])
            upper[:, -1:] += contribution(lifts[1][:, -1:], right[:, -1:], extrapolation[0][:, :1])
        else:
            diagonal[:, :1] += contribution(lifts[0][:, :1], bottom[:, None], extrapolation[0][:, :1])
            diagonal[:, -1:] += contribution(lifts[1][:, -1:], top[:, None], extrapolation[1][:, -1:])
        shape = (state_columns.shape[0], dimensions.num_elem_z, 3 * dimensions.num_solpts, 5 * dimensions.num_solpts)
        return tuple(block.reshape(shape) for block in (lower, diagonal, upper))

    def _volume_blocks(self):
        dimensions, state_columns = self.dimensions, self.state_columns
        shape = (state_columns.shape[0], dimensions.num_elem_z, 3, dimensions.num_solpts, 5, dimensions.num_solpts)
        lower, diagonal, upper = (state_columns.new_zeros(shape) for _ in range(3))
        derivative = self.derivative
        identity = torch.eye(dimensions.num_solpts, dtype=state_columns.dtype, device=state_columns.device)
        rho = state_columns[:, :, 0]

        for output, variable in ((0, 0), (2, 4)):
            flux = state_columns.new_zeros((state_columns.shape[0], dimensions.num_elem_z, 5, dimensions.num_solpts))
            flux[:, :, variable] = self.contravariant_velocity[:, :, 2]
            for j in range(3):
                flux[:, :, j + 1] += state_columns[:, :, variable] * self.inverse_spatial[2, j]
            diagonal[:, :, output] -= (
                torch.einsum("os,cejs->ceojs", derivative, self.jacobian[:, :, None] * flux)
                / self.jacobian[:, :, :, None, None]
            )

        pressure_derivative = heat_capacity_ratio * pressure(state_columns[:, :, 4]) / state_columns[:, :, 4]
        diagonal[:, :, 1, :, 4] -= torch.einsum("os,ces->ceos", derivative, pressure_derivative) / rho[:, :, :, None]
        diagonal[:, :, 1, :, 0] += identity * (self.pressure_gradient / rho.square())[:, :, :, None]
        for j in range(3):
            diagonal[:, :, 1, :, j + 1] -= torch.einsum(
                "os,ces->ceos", derivative, self.contravariant_velocity[:, :, j]
            )
            if j < 2:
                diagonal[:, :, 1, :, j + 1] += self.contravariant_velocity[:, :, j, :, None] * derivative
            local_velocity_derivative = sum(
                self.inverse_spatial[a, j]
                * (self.vertical_covariant_gradient[:, :, a] + self.vertical_frame_gradient[:, :, a])
                for a in range(2)
            )
            diagonal[:, :, 1, :, j + 1] += identity * local_velocity_derivative[:, :, :, None]

        return lower, diagonal, upper

    def _extrapolation_jacobians(self):
        """Log extrapolation gives d(q_face)/dq_node = q_face * E / q_node."""
        state_columns = self.state_columns
        extrapolation = []
        for face in range(2):
            matrix = self.extrapolation[face].expand_as(state_columns).clone()
            for variable in (0, 4):
                matrix[:, :, variable] *= self.state_trace[:, :, face, variable, None] / state_columns[:, :, variable]
            extrapolation.append(matrix)

        return extrapolation

    def _face_lifting_matrices(self):
        """Map (J flux_rho, J flux_theta, p*, K*, v_1*, v_2*) to the three equation rows."""
        dimensions, state_columns = self.dimensions, self.state_columns
        rho = state_columns[:, :, 0]
        lifts = []
        for face in range(2):
            lift = state_columns.new_zeros((state_columns.shape[0], dimensions.num_elem_z, 3, dimensions.num_solpts, 6))
            correction = self.correction[:, face]
            lift[:, :, 0, :, 0] = -correction / self.jacobian
            lift[:, :, 2, :, 1] = -correction / self.jacobian
            lift[:, :, 1, :, 2] = -correction / rho
            lift[:, :, 1, :, 3] = -correction
            lift[:, :, 1, :, 4] = correction * self.contravariant_velocity[:, :, 0]
            lift[:, :, 1, :, 5] = correction * self.contravariant_velocity[:, :, 1]
            lifts.append(lift)

        return lifts

    def __call__(self, vector):
        state = vector.reshape_as(self.state).to(self.state.dtype)
        columns = state_to_columns(self.rhs, state)
        result = torch.zeros_like(columns)
        result[self.column_slice, :, self.rows] = blocks_matvec(
            *self.blocks, columns[self.column_slice], periodic=self.rhs.periodic_vertical
        )
        return columns_to_state(self.rhs, result, state).flatten()

    def solve(self, vector, dt):
        """Solve (I-dt J1/2) delta=vector, including coupling from horizontal velocity."""
        columns = state_to_columns(self.rhs, vector.reshape_as(self.state)).clone()
        columns[self.column_slice] = self.solve_columns(columns[self.column_slice], dt)
        return columns_to_state(self.rhs, columns, self.state).reshape(self.shape)

    def solve_columns(self, columns, dt):
        """Solve the selected columns without converting a complete state."""
        columns = columns.clone()
        horizontal = columns.clone()
        horizontal[..., self.rows] = 0
        right = columns[..., self.rows] + dt / 2 * blocks_matvec(
            *self.blocks, horizontal, periodic=self.rhs.periodic_vertical
        )
        stiff = tuple(block[..., self.rows] for block in self.blocks)
        solve = cyclic_block_solve if self.rhs.periodic_vertical else block_thomas_solve
        columns[..., self.rows] = solve(*stiff, right, dt)
        return columns


class EulerPartitionLinearization:
    def __init__(self, rhs, state, *, include_explicit=True):
        self.rhs = rhs
        self.state = state
        self.full = rhs.linearize(state) if include_explicit else None
        self._implicit = None

    @property
    def implicit(self):
        if self._implicit is None:
            self._implicit = EulerVerticalLinearization(self.rhs, self.state)
        return self._implicit

    def explicit(self, vector):
        return self.full.explicit(vector)

    def release_explicit(self):
        """Release exponential-stage caches before assembling implicit blocks."""
        self.full = None

    def solve(self, vector, dt, *, column_batch_size=None):
        if self._implicit is not None and column_batch_size is None:
            return self._implicit.solve(vector, dt)
        columns = state_to_columns(self.rhs, vector.reshape_as(self.state)).clone()
        if column_batch_size is None:
            column_batch_size = 512 if vector.is_cuda else columns.shape[0]
        if column_batch_size < 1:
            raise ValueError("Column batch size must be positive")
        for start in range(0, columns.shape[0], column_batch_size):
            selection = slice(start, min(start + column_batch_size, columns.shape[0]))
            linearization = EulerVerticalLinearization(self.rhs, self.state, columns=selection)
            columns[selection] = linearization.solve_columns(columns[selection], dt)
            del linearization
        return columns_to_state(self.rhs, columns, self.state).reshape(self.state.shape)


class EulerImplicitLinearization(EulerPartitionLinearization):
    """Apply J1 or solve its column system without constructing J2."""

    def __init__(self, rhs, state):
        super().__init__(rhs, state, include_explicit=False)

    def __call__(self, vector):
        return self.implicit(vector)
