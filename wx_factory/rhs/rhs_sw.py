"""Shallow-water DFR tendency for the state (H, v_1, v_2)."""

from copy import copy

import torch

from ..common.definitions import gravity
from ..pde.shallow_water_interfaces import GravityWaveInterface


class RhsShallowWater:
    def __init__(self, geom, operators_real, metric, topo, ptopo, *, shape=None):
        self.shape = (3,) + geom.grid_shape if shape is None else shape
        self.geom = geom
        self.ops = operators_real
        self.metric = metric
        self.topo = topo
        self.ptopo = ptopo
        self.context = geom.context
        self.extrapolations = (self.ops.extrap_x, self.ops.extrap_y)
        self.derivatives = (self.ops.derivative_x, self.ops.derivative_y)
        self.corrections = (self.ops.correction_WE, self.ops.correction_SN)
        width = geom.num_solpts
        self.left = (
            (..., slice(None, -1), slice(width, None)),
            (..., slice(None, -1), slice(None), slice(width, None)),
        )
        self.right = ((..., slice(1, None), slice(None, width)), (..., slice(1, None), slice(None), slice(None, width)))
        self.interior = ((..., slice(1, -1), slice(None)), (..., slice(1, -1), slice(None), slice(None)))
        self.face_metrics = tuple(fields.at(indices) for fields, indices in zip(metric.faces, self.left))
        self.bottom = 0.0 if topo is None else topo.hsurf
        self.face_bottom = (
            (0.0, 0.0) if topo is None else (topo.hsurf_itf_i[self.left[0]], topo.hsurf_itf_j[self.left[1]])
        )

    def _start_exchange(self, state, *, perturbation=False):
        values = state.clone()
        if not perturbation:
            values[0] += self.bottom
        layers = state.shape[1:-3]
        shapes = (self.geom.itf_i_shape, self.geom.itf_j_shape)
        faces = []
        for shape, extrapolation, interior in zip(shapes, self.extrapolations, self.interior):
            face = state.new_zeros((3,) + layers + shape)
            face[interior] = values @ extrapolation
            faces.append(face)
        width = self.geom.num_solpts
        i, j = faces
        boundaries = ((j[..., 1, :, :width], j[..., -2, :, width:]), (i[..., 1, :width], i[..., -2, width:]))
        south, north = boundaries[0]
        west, east = boundaries[1]
        vector_request = self.ptopo.start_exchange_vectors(
            south=tuple(south[1:3]),
            north=tuple(north[1:3]),
            west=tuple(west[1:3]),
            east=tuple(east[1:3]),
            boundary_sn=self.geom.boundary_sn,
            boundary_we=self.geom.boundary_we,
            covariant=True,
        )
        scalar_request = self.ptopo.start_exchange_scalars(
            south=south[0],
            north=north[0],
            west=west[0],
            east=east[0],
            boundary_shape=(self.geom.num_elements_horizontal * width,),
        )
        return faces, vector_request, scalar_request

    def _finish_exchange(self, exchange):
        faces, vector_request, scalar_request = exchange
        width = self.geom.num_solpts
        i, j = faces
        destinations = (
            (j, (..., 0, slice(None), slice(width, None))),
            (j, (..., -1, slice(None), slice(None, width))),
            (i, (..., 0, slice(width, None))),
            (i, (..., -1, slice(None, width))),
        )
        vectors = vector_request.wait()
        scalars = scalar_request.wait()
        for (face, index), vector, scalar in zip(destinations, vectors, scalars):
            face[0][index] = scalar
            face[1][index], face[2][index] = vector
        return faces

    def _interfaces(self, faces):
        return tuple(
            GravityWaveInterface(face[left], face[right], fields, bottom, direction)
            for direction, (face, left, right, fields, bottom) in enumerate(
                zip(faces, self.left, self.right, self.face_metrics, self.face_bottom)
            )
        )

    def _volume_terms(self, state, velocity):
        energy = (state[1:3] * velocity).sum(dim=0) / 2 + gravity * (state[0] + self.bottom)
        return tuple(
            torch.stack((self.metric.sqrtG * state[0] * velocity[direction], energy, state[2 - direction]))
            for direction in range(2)
        )

    def _volume_derivatives(self, volume_terms):
        return tuple(value @ operator for value, operator in zip(volume_terms, self.derivatives))

    def _add_face_corrections(self, partial, common_values):
        derivatives = []
        for direction, (values, common) in enumerate(zip(partial, common_values)):
            shape = (3,) + values.shape[1:-3] + (self.geom.itf_i_shape if direction == 0 else self.geom.itf_j_shape)
            trace = values.new_zeros(shape)
            selected = common[[0, 1, 3 - direction]]
            trace[self.left[direction]] = selected
            trace[self.right[direction]] = selected
            derivatives.append(values + trace[self.interior[direction]] @ self.corrections[direction])
        return derivatives

    def __call__(self, state):
        given_shape = state.shape
        if state.ndim == 1:
            state = state.reshape(self.shape)
        exchange = self._start_exchange(state)
        velocity = self.metric.volume.raise_velocity(state[1:3])
        partial = self._volume_derivatives(self._volume_terms(state, velocity))
        interfaces = self._interfaces(self._finish_exchange(exchange))
        dx, dy = self._add_face_corrections(partial, tuple(face.common_values() for face in interfaces))
        absolute_curl = dx[2] - dy[2] + self.metric.volume.frame_vorticity
        return torch.stack(
            (
                -(dx[0] + dy[0]) / self.metric.sqrtG,
                -dx[1] + velocity[1] * absolute_curl,
                -dy[1] - velocity[0] * absolute_curl,
            )
        ).reshape(given_shape)

    def linearize(self, state):
        from ..jacobian.shallow_water import ShallowWaterLinearization

        return ShallowWaterLinearization(self, state.reshape(self.shape) if state.ndim == 1 else state)

    def double_precision_rhs(self):
        if not hasattr(self, "_double_rhs"):
            metric = copy(self.metric)
            metric.to_runtime(torch.float64)
            operators = copy(self.ops)
            for name in ("extrap_x", "extrap_y", "derivative_x", "derivative_y", "correction_WE", "correction_SN"):
                setattr(operators, name, getattr(operators, name).to(torch.float64))
            topo = None if self.topo is None else copy(self.topo)
            if topo is not None:
                for name, value in vars(topo).items():
                    if isinstance(value, torch.Tensor):
                        setattr(topo, name, value.to(torch.float64))
            self._double_rhs = RhsShallowWater(self.geom, operators, metric, topo, self.ptopo, shape=self.shape)
        return self._double_rhs

    def jacobian_rhs(self, state):
        return self.double_precision_rhs()(state.to(torch.float64))
