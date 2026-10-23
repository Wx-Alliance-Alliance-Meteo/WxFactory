"""Direct action of the shallow-water DFR Jacobian."""

import torch

from ..common.definitions import gravity


class ShallowWaterLinearization:
    def __init__(self, rhs, state):
        self.rhs = rhs
        self.state = state
        self.shape = state.shape
        self.velocity = rhs.metric.volume.raise_velocity(state[1:3])
        exchange = rhs._start_exchange(state)
        partial = rhs._volume_derivatives(rhs._volume_terms(state, self.velocity))
        self.interfaces = rhs._interfaces(rhs._finish_exchange(exchange))
        dx, dy = rhs._add_face_corrections(partial, tuple(face.common_values() for face in self.interfaces))
        self.absolute_curl = dx[2] - dy[2] + rhs.metric.volume.frame_vorticity

    def __call__(self, vector):
        rhs = self.rhs
        perturbation = vector.reshape(self.shape)
        exchange = rhs._start_exchange(perturbation, perturbation=True)
        delta_velocity = rhs.metric.volume.raise_velocity(perturbation[1:3])
        delta_energy = (self.velocity * perturbation[1:3]).sum(dim=0) + gravity * perturbation[0]
        volume_terms = tuple(
            torch.stack(
                (
                    rhs.metric.sqrtG
                    * (perturbation[0] * self.velocity[direction] + self.state[0] * delta_velocity[direction]),
                    delta_energy,
                    perturbation[2 - direction],
                )
            )
            for direction in range(2)
        )
        partial = rhs._volume_derivatives(volume_terms)
        faces = rhs._finish_exchange(exchange)
        common = tuple(
            interface.linearize(face[left], face[right])
            for interface, face, left, right in zip(self.interfaces, faces, rhs.left, rhs.right)
        )
        dx, dy = rhs._add_face_corrections(partial, common)
        delta_curl = dx[2] - dy[2]
        result = torch.stack(
            (
                -(dx[0] + dy[0]) / rhs.metric.sqrtG,
                -dx[1] + delta_velocity[1] * self.absolute_curl + self.velocity[1] * delta_curl,
                -dy[1] - delta_velocity[0] * self.absolute_curl - self.velocity[0] * delta_curl,
            )
        )
        return result.flatten()
