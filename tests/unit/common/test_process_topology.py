import unittest
import sys

import numpy
from mpi4py import MPI

from device import CpuDevice
from wx_mpi import ProcessTopology, SOUTH, NORTH, WEST, EAST

from tests.unit.mpi_test import run_test_on_x_process

dev = CpuDevice()


def gen_data_1(num_processes: int, num_data_hori_per_proc: int):
    range_per_proc = 2.0 / num_processes
    range_per_side = range_per_proc / 4
    range_per_elem = range_per_side / num_data_hori_per_proc * (1.0 + 1e-14)

    def make_range(proc, side):
        start = proc * range_per_proc + side * range_per_side - 1.0
        stop = proc * range_per_proc + (side + 1) * range_per_side - 1.0
        start += range_per_elem / 2.0
        stop += range_per_elem / 2.0
        return numpy.arange(start, stop, range_per_elem)

    data = [[make_range(proc, side) for side in range(4)] for proc in range(num_processes)]
    return numpy.array(data)


class ProcessTopologyTest(unittest.TestCase):
    topo: ProcessTopology
    rank: int
    comm: MPI.Comm

    def setUp(self) -> None:
        super().setUp()
        self.comm: MPI.Comm = run_test_on_x_process(self, 6)

        self.size = self.comm.size
        self.rank = self.comm.rank

        self.topo = ProcessTopology(dev, comm=self.comm)
        self.topos = [ProcessTopology(dev, rank=i, comm=self.comm) for i in range(self.size)]

        self.neighbor_topo = [
            self.topos[self.topo.destinations[SOUTH]],
            self.topos[self.topo.destinations[NORTH]],
            self.topos[self.topo.destinations[WEST]],
            self.topos[self.topo.destinations[EAST]],
        ]

        self.NUM_DATA_HORI = 12

        self.all_data = gen_data_1(self.size, self.NUM_DATA_HORI)
        self.coord = numpy.arange(-1.0 + 1.0 / self.NUM_DATA_HORI, 1.0, 2.0 / self.NUM_DATA_HORI)
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

    def vector2d_1d_shape1d(self):
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
                r0_other = numpy.flip(r0_other)
                r1_other = numpy.flip(r1_other)
            diff_s = numpy.linalg.norm(result[dir][0] - r0_other + result[dir][1] - r1_other)
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
                r0_other = numpy.flip(r0_other)
                r1_other = numpy.flip(r1_other)

            r0_other = r0_other.reshape(new_shape)
            r1_other = r1_other.reshape(new_shape)

            diff_s = numpy.linalg.norm(result[dir][0] - r0_other + result[dir][1] - r1_other)
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
        def make_data(d):
            return (numpy.stack([d, d + 1.0]), numpy.stack([d[::-1], d[::-1] + 1.0]))

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
                r0_other = numpy.flip(r0_other, axis=-1)
                r1_other = numpy.flip(r1_other, axis=-1)

            diff_s = numpy.linalg.norm(result[dir][0] - r0_other + result[dir][1] - r1_other)
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
        new_shape = (2, 3, 2)

        def make_data(d):
            return (
                numpy.stack([d, d + 1.0]).reshape((2,) + new_shape),
                numpy.stack([d[::-1], d[::-1] + 1.0]).reshape((2,) + new_shape),
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
                r0_other = numpy.flip(r0_other, axis=1)
                r1_other = numpy.flip(r1_other, axis=1)

            r0_other = r0_other.reshape((2,) + new_shape)
            r1_other = r1_other.reshape((2,) + new_shape)

            diff_s = numpy.linalg.norm(result[dir][0] - r0_other + result[dir][1] - r1_other)
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
                r0_other = numpy.flip(r0_other)
                r1_other = numpy.flip(r1_other)
                r2_other = numpy.flip(r2_other)

            diff_s = (
                numpy.linalg.norm(result[dir][0] - r0_other)
                + numpy.linalg.norm(result[dir][1] - r1_other)
                + numpy.linalg.norm(result[dir][2] - r2_other)
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
                r0_other = numpy.flip(r0_other)
                r1_other = numpy.flip(r1_other)
                r2_other = numpy.flip(r2_other)

            r0_other = r0_other.reshape(new_data_shape)
            r1_other = r1_other.reshape(new_data_shape)
            r2_other = r2_other.reshape(new_data_shape)

            diff_s = (
                numpy.linalg.norm(result[dir][0] - r0_other)
                + numpy.linalg.norm(result[dir][1] - r1_other)
                + numpy.linalg.norm(result[dir][2] - r2_other)
            )
            self.assertLess(
                diff_s,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff_s:.2e})\n"
                f" expected {r0_other}\n"
                f" got      {result[dir][0]}",
            )

    def vector3d_3d_shape1d(self):
        def make_data(d):
            e = d[::-1]
            return (
                numpy.array([[[d, d + 1.0], [d + 0.1, d + 1.1]], [[d + 0.2, d + 1.2], [d + 2.2, d + 3.2]]]),
                numpy.array([[[e, e + 1.0], [e + 0.1, e + 1.1]], [[e + 0.2, e + 1.2], [e + 2.2, e + 3.2]]]),
                numpy.array([[[d + 0.3, d + 1.3], [d + 0.4, d + 1.4]], [[d + 0.5, d + 1.5], [d + 2.6, d + 3.6]]]),
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
                r0_other = numpy.flip(r0_other, axis=3)
                r1_other = numpy.flip(r1_other, axis=3)
                r2_other = numpy.flip(r2_other, axis=3)

            diff_s = (
                numpy.linalg.norm(result[dir][0] - r0_other)
                + numpy.linalg.norm(result[dir][1] - r1_other)
                + numpy.linalg.norm(result[dir][2] - r2_other)
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
        base_shape = (3, 4)
        new_data_shape = (2, 2, 2, 2) + base_shape
        new_line_shape = (2, 2, 2, 2) + (self.NUM_DATA_HORI,)

        def make_data(d):
            e = d[::-1]
            return (
                numpy.array(
                    [
                        [[[d, d + 1.0], [d + 0.1, d + 1.1]], [[d + 0.2, d + 1.2], [d + 2.2, d + 3.2]]],
                        [[[d, d + 1.0], [d + 0.1, d + 1.1]], [[d + 0.2, d + 1.2], [d + 2.2, d + 3.2]]],
                    ]
                ).reshape(new_data_shape),
                numpy.array(
                    [
                        [[[e, e + 1.0], [e + 0.1, e + 1.1]], [[e + 0.2, e + 1.2], [e + 2.2, e + 3.2]]],
                        [[[e, e + 1.0], [e + 0.1, e + 1.1]], [[e + 0.2, e + 1.2], [e + 2.2, e + 3.2]]],
                    ]
                ).reshape(new_data_shape),
                numpy.array(
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
                r0_other = numpy.flip(r0_other, axis=4)
                r1_other = numpy.flip(r1_other, axis=4)
                r2_other = numpy.flip(r2_other, axis=4)

            r0_other = r0_other.reshape(new_data_shape)
            r1_other = r1_other.reshape(new_data_shape)
            r2_other = r2_other.reshape(new_data_shape)

            diff_s = (
                numpy.linalg.norm(result[dir][0] - r0_other)
                + numpy.linalg.norm(result[dir][1] - r1_other)
                + numpy.linalg.norm(result[dir][2] - r2_other)
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
                other = numpy.flip(other)
            diff = numpy.linalg.norm(result[dir] - other)
            self.assertLess(
                diff,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff:.2e})\n"
                f"expected\n{other}\n"
                f"got\n{result[dir]}",
            )

    def scalar_1d_shape2d(self):
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
                other = numpy.flip(other)
            other = other.reshape(new_shape)
            diff = numpy.linalg.norm(result[dir] - other)
            self.assertLess(
                diff,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff:.2e})\n"
                f"expected\n{other}\n"
                f"got\n{result[dir]}",
            )

    def scalar_1d_shape3d(self):
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
                other = numpy.flip(other)
            other = other.reshape(new_shape)
            diff = numpy.linalg.norm(result[dir] - other)
            self.assertLess(
                diff,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff:.2e})\n"
                f"expected\n{other}\n"
                f"got\n{result[dir]}",
            )

    def scalar_2d_shape1d(self):
        south = numpy.stack([self.data[SOUTH], self.data[SOUTH] + 1.0])
        north = numpy.stack([self.data[NORTH], self.data[NORTH] + 1.0])
        west = numpy.stack([self.data[WEST], self.data[WEST] + 1.0])
        east = numpy.stack([self.data[EAST], self.data[EAST] + 1.0])
        request = self.topo.start_exchange_scalars(south, north, west, east, boundary_shape=(self.NUM_DATA_HORI,))
        s, n, w, e = request.wait()
        result = [s, n, w, e]

        for dir in [SOUTH, NORTH, WEST, EAST]:
            other = self.neighbor_data[dir][self.from_neighbor[dir]]
            if self.neighbor_topo[dir].flip[self.from_neighbor[dir]]:
                other = numpy.flip(other)

            other = numpy.stack([other, other + 1.0])
            diff = numpy.linalg.norm(result[dir] - other)
            self.assertLess(
                diff,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff:.2e})\n"
                f"expected\n{other}\n"
                f"got\n{result[dir]}",
            )

    def scalar_2d_shape2d(self):
        new_shape = (4, 3)
        south = numpy.stack([self.data[SOUTH].reshape(new_shape), self.data[SOUTH].reshape(new_shape) + 1.0])
        north = numpy.stack([self.data[NORTH].reshape(new_shape), self.data[NORTH].reshape(new_shape) + 1.0])
        west = numpy.stack([self.data[WEST].reshape(new_shape), self.data[WEST].reshape(new_shape) + 1.0])
        east = numpy.stack([self.data[EAST].reshape(new_shape), self.data[EAST].reshape(new_shape) + 1.0])
        request = self.topo.start_exchange_scalars(south, north, west, east, boundary_shape=(self.NUM_DATA_HORI,))
        s, n, w, e = request.wait()
        result = [s, n, w, e]

        for dir in [SOUTH, NORTH, WEST, EAST]:
            other = self.neighbor_data[dir][self.from_neighbor[dir]]
            if self.neighbor_topo[dir].flip[self.from_neighbor[dir]]:
                other = numpy.flip(other)

            other = other.reshape(new_shape)
            other = numpy.stack([other, other + 1.0])
            diff = numpy.linalg.norm(result[dir] - other)
            self.assertLess(
                diff,
                1e-15,
                f"rank {self.rank}: {dir} data is wrong (norm {diff:.2e})\n"
                f"expected\n{other}\n"
                f"got\n{result[dir]}",
            )
