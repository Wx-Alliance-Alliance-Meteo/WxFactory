import unittest
import sys

import numpy
from mpi4py import MPI

from device import Device, CpuDevice, CudaDevice
from process_topology import ProcessTopology, SOUTH, NORTH, WEST, EAST
from wx_mpi import SingleProcess, Conditional

from tests.unit.mpi_test import run_test_on_x_process, MpiTestCase

TestDeviceClass = CpuDevice


def gen_data_1(num_processes: int, num_data_hori_per_proc: int, device: Device):
    xp = device.xp

    range_per_proc = 2.0 / num_processes
    range_per_side = range_per_proc / 4
    range_per_elem = range_per_side / num_data_hori_per_proc * (1.0 + 1e-14)

    def make_range(proc, side):
        start = proc * range_per_proc + side * range_per_side - 1.0
        stop = proc * range_per_proc + (side + 1) * range_per_side - 1.0
        start += range_per_elem / 2.0
        stop += range_per_elem / 2.0
        return xp.arange(start, stop, range_per_elem)

    data = [[make_range(proc, side) for side in range(4)] for proc in range(num_processes)]
    return xp.array(data)


class ExchangeTest(unittest.TestCase):
    topo: ProcessTopology
    rank: int
    comm: MPI.Comm

    def setUp(self) -> None:
        super().setUp()
        self.comm: MPI.Comm = run_test_on_x_process(self, 6)

        self.size = self.comm.size
        self.rank = self.comm.rank

        dev = TestDeviceClass(self.comm)

        self.topo = ProcessTopology(dev, comm_in=self.comm)
        self.topos = [ProcessTopology(dev, rank=i, comm_in=self.comm) for i in range(self.size)]

        self.neighbor_topo = [
            self.topos[self.topo.destinations[SOUTH]],
            self.topos[self.topo.destinations[NORTH]],
            self.topos[self.topo.destinations[WEST]],
            self.topos[self.topo.destinations[EAST]],
        ]

        self.NUM_DATA_HORI = 12

        self.all_data = gen_data_1(self.size, self.NUM_DATA_HORI, dev)
        self.coord = dev.xp.arange(-1.0 + 1.0 / self.NUM_DATA_HORI, 1.0, 2.0 / self.NUM_DATA_HORI)
        # if self.rank == 0:
        #     print(f'coord = {self.coord}')

        # print(f'rank {self.rank}:\n'
        #       f'data_south = {self.data[self.rank][0]}\n'
        #       f'data_north = {self.data[self.rank][1]}\n'
        #       f'data_west  = {self.data[self.rank][2]}\n'
        #       f'data_east  = {self.data[self.rank][3]}')

        self.to_neighbor = self.topo.destinations
        self.from_neighbor = [-1, -1, -1, -1]
        for i in range(4):
            for j in range(4):
                if self.neighbor_topo[i].sources[j] == self.rank:
                    self.from_neighbor[i] = j

        self.data = self.all_data[self.rank]
        self.neighbor_data = [self.all_data[x] for x in self.to_neighbor]
        self.xp = dev.xp

    def vector2d_1d_shape1d(self):
        xp = self.xp
        south = (self.data[SOUTH], self.data[SOUTH][::-1])
        north = (self.data[NORTH], self.data[NORTH][::-1])
        west = (self.data[WEST], self.data[WEST][::-1])
        east = (self.data[EAST], self.data[EAST][::-1])
        request = self.topo.start_exchange_vectors(south, north, west, east, self.coord, self.coord)
        (s1, s2), (n1, n2), (w1, w2), (e1, e2) = request.wait()
        result = [(s1, s2), (n1, n2), (w1, w2), (e1, e2)]

        sys.stdout.flush()

        for dir in [SOUTH, NORTH, WEST, EAST]:
            other0 = self.neighbor_data[dir][self.from_neighbor[dir]]
            other1 = other0[::-1]
            r0_other, r1_other = self.neighbor_topo[dir].convert_contra[self.from_neighbor[dir]](
                other0, other1, self.coord
            )
            if self.neighbor_topo[dir].flip[self.from_neighbor[dir]]:
                r0_other = xp.flip(r0_other)
                r1_other = xp.flip(r1_other)
            diff_s = xp.linalg.norm(result[dir][0] - r0_other + result[dir][1] - r1_other)
            self.assertLess(
                diff_s,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff_s:.2e})\n"
                f" expected {r0_other}\n"
                f"          {r1_other}\n"
                f" got      {result[dir][0]}\n"
                f"          {result[dir][1]}",
            )

    def vector2d_1d_shape2d(self):
        xp = self.xp
        new_shape = (6, 2)
        south = (self.data[SOUTH].reshape(new_shape), self.data[SOUTH][::-1].reshape(new_shape))
        north = (self.data[NORTH].reshape(new_shape), self.data[NORTH][::-1].reshape(new_shape))
        west = (self.data[WEST].reshape(new_shape), self.data[WEST][::-1].reshape(new_shape))
        east = (self.data[EAST].reshape(new_shape), self.data[EAST][::-1].reshape(new_shape))
        request = self.topo.start_exchange_vectors(south, north, west, east, self.coord, self.coord)
        (s1, s2), (n1, n2), (w1, w2), (e1, e2) = request.wait()
        result = [(s1, s2), (n1, n2), (w1, w2), (e1, e2)]

        sys.stdout.flush()

        for dir in [SOUTH, NORTH, WEST, EAST]:
            other0 = self.neighbor_data[dir][self.from_neighbor[dir]]
            other1 = other0[::-1]
            r0_other, r1_other = self.neighbor_topo[dir].convert_contra[self.from_neighbor[dir]](
                other0, other1, self.coord
            )
            if self.neighbor_topo[dir].flip[self.from_neighbor[dir]]:
                r0_other = xp.flip(r0_other)
                r1_other = xp.flip(r1_other)

            r0_other = r0_other.reshape(new_shape)
            r1_other = r1_other.reshape(new_shape)

            diff_s = xp.linalg.norm(result[dir][0] - r0_other + result[dir][1] - r1_other)
            self.assertLess(
                diff_s,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff_s:.2e})\n"
                f" expected \n"
                f"{r0_other}\n"
                f"{r1_other}\n"
                f"got\n"
                f"{result[dir][0]}\n"
                f"{result[dir][1]}",
            )

    def vector2d_2d_shape1d(self):
        xp = self.xp

        def make_data(d):
            return (xp.stack([d, d + 1.0]), xp.stack([d[::-1], d[::-1] + 1.0]))

        south = make_data(self.data[SOUTH])
        north = make_data(self.data[NORTH])
        west = make_data(self.data[WEST])
        east = make_data(self.data[EAST])

        request = self.topo.start_exchange_vectors(south, north, west, east, self.coord, self.coord)
        (s1, s2), (n1, n2), (w1, w2), (e1, e2) = request.wait()
        result = [(s1, s2), (n1, n2), (w1, w2), (e1, e2)]

        sys.stdout.flush()

        for dir in [SOUTH, NORTH, WEST, EAST]:
            other = make_data(self.neighbor_data[dir][self.from_neighbor[dir]])
            r0_other, r1_other = self.neighbor_topo[dir].convert_contra[self.from_neighbor[dir]](
                other[0], other[1], self.coord
            )
            if self.neighbor_topo[dir].flip[self.from_neighbor[dir]]:
                r0_other = xp.flip(r0_other, axis=-1)
                r1_other = xp.flip(r1_other, axis=-1)

            diff_s = xp.linalg.norm(result[dir][0] - r0_other + result[dir][1] - r1_other)
            self.assertLess(
                diff_s,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff_s:.2e})\n"
                f" expected \n"
                f"{r0_other}\n"
                f"{r1_other}\n"
                f"got\n"
                f"{result[dir][0]}\n"
                f"{result[dir][1]}",
            )

    def vector2d_2d_shape3d(self):
        xp = self.xp
        new_shape = (2, 3, 2)

        def make_data(d):
            return (
                xp.stack([d, d + 1.0]).reshape((2,) + new_shape),
                xp.stack([d[::-1], d[::-1] + 1.0]).reshape((2,) + new_shape),
            )

        south = make_data(self.data[SOUTH])
        north = make_data(self.data[NORTH])
        west = make_data(self.data[WEST])
        east = make_data(self.data[EAST])

        request = self.topo.start_exchange_vectors(south, north, west, east, self.coord, self.coord)
        (s1, s2), (n1, n2), (w1, w2), (e1, e2) = request.wait()
        result = [(s1, s2), (n1, n2), (w1, w2), (e1, e2)]

        sys.stdout.flush()

        for dir in [SOUTH, NORTH, WEST, EAST]:
            other = make_data(self.neighbor_data[dir][self.from_neighbor[dir]])
            r0_other, r1_other = self.neighbor_topo[dir].convert_contra[self.from_neighbor[dir]](
                other[0].reshape((2, self.NUM_DATA_HORI)), other[1].reshape((2, self.NUM_DATA_HORI)), self.coord
            )
            if self.neighbor_topo[dir].flip[self.from_neighbor[dir]]:
                r0_other = xp.flip(r0_other, axis=1)
                r1_other = xp.flip(r1_other, axis=1)

            r0_other = r0_other.reshape((2,) + new_shape)
            r1_other = r1_other.reshape((2,) + new_shape)

            diff_s = xp.linalg.norm(result[dir][0] - r0_other + result[dir][1] - r1_other)
            self.assertLess(
                diff_s,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff_s:.2e})\n"
                f" expected \n"
                f"{r0_other}\n"
                f"{r1_other}\n"
                f"got\n"
                f"{result[dir][0]}\n"
                f"{result[dir][1]}",
            )

    def vector3d_1d_shape1d(self):
        xp = self.xp

        def make_data(d):
            return (d, d[::-1], d + 5.0)

        south = make_data(self.data[SOUTH])
        north = make_data(self.data[NORTH])
        west = make_data(self.data[WEST])
        east = make_data(self.data[EAST])
        request = self.topo.start_exchange_vectors(south, north, west, east, self.coord, self.coord)
        (s1, s2, s3), (n1, n2, n3), (w1, w2, w3), (e1, e2, e3) = request.wait()
        result = [(s1, s2, s3), (n1, n2, n3), (w1, w2, w3), (e1, e2, e3)]

        sys.stdout.flush()

        for dir in [SOUTH, NORTH, WEST, EAST]:
            other = make_data(self.neighbor_data[dir][self.from_neighbor[dir]])
            r0_other, r1_other = self.neighbor_topo[dir].convert_contra[self.from_neighbor[dir]](
                other[0], other[1], self.coord
            )
            r2_other = other[2]
            if self.neighbor_topo[dir].flip[self.from_neighbor[dir]]:
                r0_other = xp.flip(r0_other)
                r1_other = xp.flip(r1_other)
                r2_other = xp.flip(r2_other)

            diff_s = (
                xp.linalg.norm(result[dir][0] - r0_other)
                + xp.linalg.norm(result[dir][1] - r1_other)
                + xp.linalg.norm(result[dir][2] - r2_other)
            )
            self.assertLess(
                diff_s,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff_s:.2e})\n"
                f" expected {r0_other}\n"
                f"          {r1_other}\n"
                f" got      {result[dir][0]}\n"
                f"          {result[dir][1]}",
            )

    def vector3d_1d_shape2d(self):
        xp = self.xp
        base_shape = (3, 4)
        new_data_shape = (1,) + base_shape
        new_line_shape = (1,) + (self.NUM_DATA_HORI,)

        def make_data(d):
            return (d.reshape(new_data_shape), d[::-1].reshape(new_data_shape), d.reshape(new_data_shape) + 5.0)

        south = make_data(self.data[SOUTH])
        north = make_data(self.data[NORTH])
        west = make_data(self.data[WEST])
        east = make_data(self.data[EAST])
        request = self.topo.start_exchange_vectors(south, north, west, east, self.coord, self.coord)
        (s1, s2, s3), (n1, n2, n3), (w1, w2, w3), (e1, e2, e3) = request.wait()
        result = [(s1, s2, s3), (n1, n2, n3), (w1, w2, w3), (e1, e2, e3)]

        sys.stdout.flush()

        for dir in [SOUTH, NORTH, WEST, EAST]:
            other = make_data(self.neighbor_data[dir][self.from_neighbor[dir]])
            r0_other, r1_other = self.neighbor_topo[dir].convert_contra[self.from_neighbor[dir]](
                other[0].reshape(new_line_shape), other[1].reshape(new_line_shape), self.coord
            )
            r2_other = other[2]
            if self.neighbor_topo[dir].flip[self.from_neighbor[dir]]:
                r0_other = xp.flip(r0_other)
                r1_other = xp.flip(r1_other)
                r2_other = xp.flip(r2_other)

            r0_other = r0_other.reshape(new_data_shape)
            r1_other = r1_other.reshape(new_data_shape)
            r2_other = r2_other.reshape(new_data_shape)

            diff_s = (
                xp.linalg.norm(result[dir][0] - r0_other)
                + xp.linalg.norm(result[dir][1] - r1_other)
                + xp.linalg.norm(result[dir][2] - r2_other)
            )
            self.assertLess(
                diff_s,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff_s:.2e})\n"
                f" expected {r0_other}\n"
                f" got      {result[dir][0]}",
            )

    def vector3d_3d_shape1d(self):
        xp = self.xp

        def make_data(d):
            e = d[::-1]
            return (
                xp.array([[[d, d + 1.0], [d + 0.1, d + 1.1]], [[d + 0.2, d + 1.2], [d + 2.2, d + 3.2]]]),
                xp.array([[[e, e + 1.0], [e + 0.1, e + 1.1]], [[e + 0.2, e + 1.2], [e + 2.2, e + 3.2]]]),
                xp.array([[[d + 0.3, d + 1.3], [d + 0.4, d + 1.4]], [[d + 0.5, d + 1.5], [d + 2.6, d + 3.6]]]),
            )

        south = make_data(self.data[SOUTH])
        north = make_data(self.data[NORTH])
        west = make_data(self.data[WEST])
        east = make_data(self.data[EAST])
        request = self.topo.start_exchange_vectors(south, north, west, east, self.coord, self.coord)
        (s1, s2, s3), (n1, n2, n3), (w1, w2, w3), (e1, e2, e3) = request.wait()
        result = [(s1, s2, s3), (n1, n2, n3), (w1, w2, w3), (e1, e2, e3)]

        sys.stdout.flush()

        for dir in [SOUTH, NORTH, WEST, EAST]:
            other = make_data(self.neighbor_data[dir][self.from_neighbor[dir]])
            r0_other, r1_other = self.neighbor_topo[dir].convert_contra[self.from_neighbor[dir]](
                other[0], other[1], self.coord
            )
            r2_other = other[2]
            if self.neighbor_topo[dir].flip[self.from_neighbor[dir]]:
                r0_other = xp.flip(r0_other, axis=3)
                r1_other = xp.flip(r1_other, axis=3)
                r2_other = xp.flip(r2_other, axis=3)

            diff_s = (
                xp.linalg.norm(result[dir][0] - r0_other)
                + xp.linalg.norm(result[dir][1] - r1_other)
                + xp.linalg.norm(result[dir][2] - r2_other)
            )
            self.assertLess(
                diff_s,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff_s:.2e})\n"
                f" expected \n{r0_other}\n"
                f"{r1_other}\n"
                f" got      \n{result[dir][0]}\n"
                f"{result[dir][1]}",
            )

    def vector3d_4d_shape3d(self):
        xp = self.xp
        base_shape = (3, 4)
        new_data_shape = (2, 2, 2, 2) + base_shape
        new_line_shape = (2, 2, 2, 2) + (self.NUM_DATA_HORI,)

        def make_data(d):
            e = d[::-1]
            return (
                xp.array(
                    [
                        [[[d, d + 1.0], [d + 0.1, d + 1.1]], [[d + 0.2, d + 1.2], [d + 2.2, d + 3.2]]],
                        [[[d, d + 1.0], [d + 0.1, d + 1.1]], [[d + 0.2, d + 1.2], [d + 2.2, d + 3.2]]],
                    ]
                ).reshape(new_data_shape),
                xp.array(
                    [
                        [[[e, e + 1.0], [e + 0.1, e + 1.1]], [[e + 0.2, e + 1.2], [e + 2.2, e + 3.2]]],
                        [[[e, e + 1.0], [e + 0.1, e + 1.1]], [[e + 0.2, e + 1.2], [e + 2.2, e + 3.2]]],
                    ]
                ).reshape(new_data_shape),
                xp.array(
                    [
                        [[[d + 0.3, d + 1.3], [d + 0.4, d + 1.4]], [[d + 0.5, d + 1.5], [d + 2.6, d + 3.6]]],
                        [[[d + 0.3, d + 1.3], [d + 0.4, d + 1.4]], [[d + 0.5, d + 1.5], [d + 2.6, d + 3.6]]],
                    ]
                ).reshape(new_data_shape),
            )

        south = make_data(self.data[SOUTH])
        north = make_data(self.data[NORTH])
        west = make_data(self.data[WEST])
        east = make_data(self.data[EAST])
        request = self.topo.start_exchange_vectors(south, north, west, east, self.coord, self.coord)
        (s1, s2, s3), (n1, n2, n3), (w1, w2, w3), (e1, e2, e3) = request.wait()
        result = [(s1, s2, s3), (n1, n2, n3), (w1, w2, w3), (e1, e2, e3)]

        sys.stdout.flush()

        for dir in [SOUTH, NORTH, WEST, EAST]:
            other = make_data(self.neighbor_data[dir][self.from_neighbor[dir]])
            r0_other, r1_other = self.neighbor_topo[dir].convert_contra[self.from_neighbor[dir]](
                other[0].reshape(new_line_shape), other[1].reshape(new_line_shape), self.coord
            )
            r2_other = other[2].reshape(new_line_shape)
            if self.neighbor_topo[dir].flip[self.from_neighbor[dir]]:
                r0_other = xp.flip(r0_other, axis=4)
                r1_other = xp.flip(r1_other, axis=4)
                r2_other = xp.flip(r2_other, axis=4)

            r0_other = r0_other.reshape(new_data_shape)
            r1_other = r1_other.reshape(new_data_shape)
            r2_other = r2_other.reshape(new_data_shape)

            diff_s = (
                xp.linalg.norm(result[dir][0] - r0_other)
                + xp.linalg.norm(result[dir][1] - r1_other)
                + xp.linalg.norm(result[dir][2] - r2_other)
            )
            self.assertLess(
                diff_s,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff_s:.2e})\n"
                f" expected \n{r2_other}\n"
                f" got      \n{result[dir][2]}\n"
                f"diff \n{r2_other - result[dir][2]}",
            )

    def scalar_1d_shape1d(self):
        xp = self.xp
        south = self.data[SOUTH]
        north = self.data[NORTH]
        west = self.data[WEST]
        east = self.data[EAST]
        request = self.topo.start_exchange_scalars(south, north, west, east, boundary_shape=(self.NUM_DATA_HORI,))
        s, n, w, e = request.wait()
        result = [s, n, w, e]

        for dir in [SOUTH, NORTH, WEST, EAST]:
            other = self.neighbor_data[dir][self.from_neighbor[dir]]
            if self.neighbor_topo[dir].flip[self.from_neighbor[dir]]:
                other = xp.flip(other)
            diff = xp.linalg.norm(result[dir] - other)
            self.assertLess(
                diff,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff:.2e})\n"
                f"expected\n{other}\n"
                f"got\n{result[dir]}",
            )

    def scalar_1d_shape2d(self):
        xp = self.xp
        new_shape = (4, 3)
        south = self.data[SOUTH].reshape(new_shape)
        north = self.data[NORTH].reshape(new_shape)
        west = self.data[WEST].reshape(new_shape)
        east = self.data[EAST].reshape(new_shape)
        request = self.topo.start_exchange_scalars(south, north, west, east, boundary_shape=(self.NUM_DATA_HORI,))
        s, n, w, e = request.wait()
        result = [s, n, w, e]

        for dir in [SOUTH, NORTH, WEST, EAST]:
            other = self.neighbor_data[dir][self.from_neighbor[dir]]
            if self.neighbor_topo[dir].flip[self.from_neighbor[dir]]:
                other = xp.flip(other)
            other = other.reshape(new_shape)
            diff = xp.linalg.norm(result[dir] - other)
            self.assertLess(
                diff,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff:.2e})\n"
                f"expected\n{other}\n"
                f"got\n{result[dir]}",
            )

    def scalar_1d_shape3d(self):
        xp = self.xp
        new_shape = (2, 3, 2)
        south = self.data[SOUTH].reshape(new_shape)
        north = self.data[NORTH].reshape(new_shape)
        west = self.data[WEST].reshape(new_shape)
        east = self.data[EAST].reshape(new_shape)
        request = self.topo.start_exchange_scalars(south, north, west, east, boundary_shape=(self.NUM_DATA_HORI,))
        s, n, w, e = request.wait()
        result = [s, n, w, e]

        for dir in [SOUTH, NORTH, WEST, EAST]:
            other = self.neighbor_data[dir][self.from_neighbor[dir]]
            if self.neighbor_topo[dir].flip[self.from_neighbor[dir]]:
                other = xp.flip(other)
            other = other.reshape(new_shape)
            diff = xp.linalg.norm(result[dir] - other)
            self.assertLess(
                diff,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff:.2e})\n"
                f"expected\n{other}\n"
                f"got\n{result[dir]}",
            )

    def scalar_2d_shape1d(self):
        xp = self.xp
        south = xp.stack([self.data[SOUTH], self.data[SOUTH] + 1.0])
        north = xp.stack([self.data[NORTH], self.data[NORTH] + 1.0])
        west = xp.stack([self.data[WEST], self.data[WEST] + 1.0])
        east = xp.stack([self.data[EAST], self.data[EAST] + 1.0])
        request = self.topo.start_exchange_scalars(south, north, west, east, boundary_shape=(self.NUM_DATA_HORI,))
        s, n, w, e = request.wait()
        result = [s, n, w, e]

        for dir in [SOUTH, NORTH, WEST, EAST]:
            other = self.neighbor_data[dir][self.from_neighbor[dir]]
            if self.neighbor_topo[dir].flip[self.from_neighbor[dir]]:
                other = xp.flip(other)

            other = xp.stack([other, other + 1.0])
            diff = xp.linalg.norm(result[dir] - other)
            self.assertLess(
                diff,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff:.2e})\n"
                f"expected\n{other}\n"
                f"got\n{result[dir]}",
            )

    def scalar_2d_shape2d(self):
        xp = self.xp
        new_shape = (4, 3)
        south = xp.stack([self.data[SOUTH].reshape(new_shape), self.data[SOUTH].reshape(new_shape) + 1.0])
        north = xp.stack([self.data[NORTH].reshape(new_shape), self.data[NORTH].reshape(new_shape) + 1.0])
        west = xp.stack([self.data[WEST].reshape(new_shape), self.data[WEST].reshape(new_shape) + 1.0])
        east = xp.stack([self.data[EAST].reshape(new_shape), self.data[EAST].reshape(new_shape) + 1.0])
        request = self.topo.start_exchange_scalars(south, north, west, east, boundary_shape=(self.NUM_DATA_HORI,))
        s, n, w, e = request.wait()
        result = [s, n, w, e]

        for dir in [SOUTH, NORTH, WEST, EAST]:
            other = self.neighbor_data[dir][self.from_neighbor[dir]]
            if self.neighbor_topo[dir].flip[self.from_neighbor[dir]]:
                other = xp.flip(other)

            other = other.reshape(new_shape)
            other = xp.stack([other, other + 1.0])
            diff = xp.linalg.norm(result[dir] - other)
            self.assertLess(
                diff,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff:.2e})\n"
                f"expected\n{other}\n"
                f"got\n{result[dir]}",
            )


class GatherScatterTest(MpiTestCase):
    topo: ProcessTopology

    def __init__(self, num_procs, methodName, optional=False):
        super().__init__(num_procs, methodName, optional)

    def setUp(self) -> None:
        super().setUp()

        dev = TestDeviceClass(self.comm)
        xp = dev.xp
        self.topo = ProcessTopology(dev, comm_in=self.comm)
        # For testing gather/scatter functions
        self.global_data_1 = xp.arange(6 * 12 * 12).reshape(6, 12, 12)  # A flat (2D) field
        # A 2D field of 3x3 elements
        self.global_data_2 = xp.arange(6 * 12 * 12 * 3 * 3).reshape(6, 12, 12, 3, 3)
        # A 3D field of scalars
        self.global_data_3a = xp.arange(6 * 4 * 12 * 12).reshape(6, 4, 12, 12)
        # A 3D field of 3x3 elements
        self.global_data_3b = xp.arange(6 * 4 * 12 * 12 * 2 * 2).reshape(6, 4, 12, 12, 2, 2)
        # A 4D field of 3x3 elements
        self.global_data_4 = xp.arange(6 * 3 * 4 * 12 * 12 * 2 * 2).reshape(6, 3, 4, 12, 12, 2, 2)

        self.global_data_fail_1 = xp.arange(6 * 13 * 13).reshape(6, 13, 13)
        self.global_data_fail_2 = xp.arange(6 * 12 * 14).reshape(6, 12, 14)
        self.global_data_fail_3 = xp.arange(4 * 12 * 12).reshape(4, 12, 12)
        self.global_data_fail_4 = xp.arange(6 * 6).reshape(6, 6)
        self.xp = dev.xp

    def gather_scatter(self, global_data, num_dim):
        xp = self.xp
        side = self.topo.num_lines_per_panel
        tile_side = global_data.shape[num_dim - 1] // side
        my_panel = self.topo.my_panel
        my_row = self.topo.my_row
        my_col = self.topo.my_col
        if num_dim == 2:
            tile_data_ref = global_data[
                my_panel, my_row * tile_side : (my_row + 1) * tile_side, my_col * tile_side : (my_col + 1) * tile_side
            ]
        elif num_dim == 3:
            tile_data_ref = global_data[
                my_panel,
                :,
                my_row * tile_side : (my_row + 1) * tile_side,
                my_col * tile_side : (my_col + 1) * tile_side,
            ]
        elif num_dim == 4:
            tile_data_ref = global_data[
                my_panel,
                :,
                :,
                my_row * tile_side : (my_row + 1) * tile_side,
                my_col * tile_side : (my_col + 1) * tile_side,
            ]
        else:
            raise ValueError(f"Unhandled num dims {num_dim}. Fix the test!")

        # if self.comm.rank == 0:
        #     print(f"tile ({my_panel}, ({my_row}, {my_col})): \n{tile_data_ref}", flush=True)

        cube = self.topo.gather_cube(tile_data_ref.copy(), num_dim)
        with SingleProcess(self.topo._comm) as s, Conditional(s):
            # print(f"cube = \n{cube[0]}", flush=True)
            diff = cube - global_data
            diff_norm = xp.linalg.norm(diff)
            self.assertEqual(diff_norm, 0, f"Gathering failed")

        tile = self.topo.distribute_cube(cube, num_dim)
        tile_diff = tile_data_ref - tile
        tile_diff_norm = xp.linalg.norm(tile_diff)
        self.assertEqual(tile_diff_norm, 0, f"Distributing failed")

    def gather_scatter_2d(self):
        self.gather_scatter(self.global_data_1, 2)

    def gather_scatter_elem_2d(self):
        self.gather_scatter(self.global_data_2, 2)

    def gather_scatter_3d(self):
        self.gather_scatter(self.global_data_3a, 3)

    def gather_scatter_elem_3d(self):
        self.gather_scatter(self.global_data_3b, 3)

    def gather_scatter_elem_4d(self):
        self.gather_scatter(self.global_data_4, 4)

    @unittest.expectedFailure
    def fail_wrong_num_proc(self):
        self.topo.distribute_cube(self.global_data_fail_1, num_dim=2)

    @unittest.expectedFailure
    def fail_not_square(self):
        self.topo.distribute_cube(self.global_data_fail_2, num_dim=2)

    @unittest.expectedFailure
    def fail_not_cube(self):
        self.topo.distribute_cube(self.global_data_fail_3, num_dim=2)

    @unittest.expectedFailure
    def fail_wrong_num_dim(self):
        self.topo.distribute_cube(self.global_data_fail_4, num_dim=2)
