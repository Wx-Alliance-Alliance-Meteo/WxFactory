import torch

from tests.unit.mpi_test import MpiTestCase
from wx_factory.common import Configuration, load_default_schema, readfile
from wx_factory.geometry import CubedSphere3D, DFROperators
from wx_factory.geometry.metric5d import Metric5D
from wx_factory.process_topology import ProcessTopology


class Metric5DSeamTestCase(MpiTestCase):
    def __init__(self, num_procs, methodName="runTest", device_name="cpu", optional=False, depth="shallow"):
        super().__init__(num_procs, methodName, device_name, optional)
        self.depth = depth

    def __str__(self):
        return super().__str__() + f".{self.depth}"

    def setUp(self):
        super().setUp()
        self.config = Configuration(readfile("config/dcmip201_ROSEXP.ini"), load_default_schema())
        self.config.depth_approx = self.depth
        self.topology = ProcessTopology(self.context)
        self.geom = CubedSphere3D(
            2, 2, 3, 2 * self.topology.num_lines_per_panel, 0.2, 0.1, -0.15, 12000.0, self.topology, self.config
        )
        self.metric = Metric5D(self.geom, self.config)
        self.metric.build_metric()

    def test_vector_and_covector_exchange_over_terrain(self):
        """All seams, including panel-2 south, preserve an independently supplied Cartesian vector."""
        geom, metric = self.geom, self.metric
        width = geom.itf_size
        points = (
            (metric.faces[1], (..., 1, slice(None), slice(None, width))),
            (metric.faces[1], (..., -2, slice(None), slice(width, None))),
            (metric.faces[0], (..., 1, slice(None, width))),
            (metric.faces[0], (..., -2, slice(width, None))),
        )
        covectors, vectors, expected_physical, physical_bases = [], [], [], []
        for fields, face in points:
            # Spatially varying data exercises point-order reversals as well as rotations.
            radial = fields.radial[face]
            physical = torch.stack((1 + radial[1], -2 + radial[2], 0.7 + radial[0]))
            basis = fields.physical_basis[face]
            covector = torch.einsum("ai...,a...->i...", basis, physical)
            vector = torch.einsum("ij...,j...->i...", fields.G_contra[1:4, 1:4][face], covector)
            covectors.append(tuple(covector))
            vectors.append(tuple(vector))
            physical_bases.append(basis)
            expected_physical.append(physical)

        # Complete all exchanges before asserting on any rank.
        arguments = (geom.boundary_sn_new, geom.boundary_we_new)
        received_vectors = self.topology.start_exchange_vectors(*vectors, *arguments, flip_dim=(-3, -1)).wait()
        received_covectors = self.topology.start_exchange_vectors(
            *covectors, *arguments, flip_dim=(-3, -1), covariant=True
        ).wait()
        payloads = [
            torch.stack((physical[0], *covector, physical[2]))
            for physical, covector in zip(expected_physical, covectors)
        ]
        received_states = self.topology.start_exchange_euler_3d(
            *payloads, *arguments, flip_dim=(-3, -1), covariant=True
        ).wait()
        for index, (fields, face) in enumerate(points):
            vector = received_vectors[index]
            covector = received_covectors[index]
            physical = torch.einsum("ai...,i...->a...", physical_bases[index], vector)
            torch.testing.assert_close(physical, expected_physical[index], atol=2e-11, rtol=2e-11)
            lowered = torch.einsum("ij...,j...->i...", fields.G_cov[1:4, 1:4][face], vector)
            torch.testing.assert_close(covector, lowered, atol=2e-8, rtol=2e-11)
            torch.testing.assert_close(received_states[index][1:4], covector, rtol=0, atol=0)
            torch.testing.assert_close(received_states[index][0], expected_physical[index][0])
            torch.testing.assert_close(received_states[index][4], expected_physical[index][2])

    def test_analytic_height_matches_existing_case_variant(self):
        from wx_factory.init.dcmip import dcmip_steady_state_mountain

        ops = DFROperators(self.geom, self.context)
        self.assertFalse(hasattr(ops, "highfilter"))
        self.assertFalse(hasattr(ops, "highfilter_k"))
        dcmip_steady_state_mountain(self.geom, self.metric, ops, self.config)
        torch.testing.assert_close(self.metric.volume.height, self.geom.height_new, atol=1e-9, rtol=2e-13)

    def test_static_rebuild_is_bitwise_identical(self):
        from dataclasses import fields

        from wx_factory.geometry.metric5d import Metric5DFields

        expected = [
            {field.name: getattr(collection, field.name).clone() for field in fields(Metric5DFields)}
            for collection in (self.metric.volume, *self.metric.faces)
        ]
        self.metric.build_metric()
        for collection, reference in zip((self.metric.volume, *self.metric.faces), expected):
            for name, value in reference.items():
                self.assertTrue(torch.equal(getattr(collection, name), value), name)

    def test_reference_nodal_derivatives(self):
        ops = DFROperators(self.geom, self.context)
        nodes = self.geom.solutionPoints
        z, y, x = torch.meshgrid(nodes, nodes, nodes, indexing="ij")
        polynomial = (x * x + 2 * y * z + y * y * z).flatten()
        for derivative, extrapolate, correction, expected in (
            (ops.derivative_x, ops.extrap_x, ops.correction_WE, 2 * x),
            (ops.derivative_y, ops.extrap_y, ops.correction_SN, 2 * z + 2 * y * z),
            (ops.derivative_z, ops.extrap_z, ops.correction_DU, 2 * y + y * y),
        ):
            actual = polynomial @ derivative + (polynomial @ extrapolate) @ correction
            torch.testing.assert_close(actual, expected.flatten(), atol=1e-14, rtol=1e-13)
