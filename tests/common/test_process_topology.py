from typing import List
import unittest
import sys

import numpy
from mpi4py import MPI

from common.process_topology import ProcessTopology, SOUTH, NORTH, WEST, EAST
from common.device import CpuDevice

from tests.mpi_test import run_test_on_x_process

dev = CpuDevice()

def gen_data_1(num_processes: int, num_data_hori_per_proc: int):
    range_per_proc = 2.0 / num_processes
    range_per_side = range_per_proc / 4
    range_per_elem = range_per_side / num_data_hori_per_proc * (1.0 + 1e-14)

    def make_range(proc, side):
        start = proc*range_per_proc + side*range_per_side - 1.0
        stop  = proc*range_per_proc + (side+1)*range_per_side - 1.0
        start += range_per_elem / 2.0
        stop  += range_per_elem / 2.0
        return numpy.arange(start, stop, range_per_elem)

    data = [[make_range(proc, side) for side in range(4)] for proc in range(num_processes)]
    return data

class ProcessTopologyTest(unittest.TestCase):
    topo: ProcessTopology
    rank: int
    data: List[List[numpy.ndarray]]
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
            self.topos[self.topo.destinations[EAST]]
        ]

        NUM_DATA_HORI = 5

        self.all_data = gen_data_1(self.size, NUM_DATA_HORI)
        self.coord = numpy.arange(-1.0 + 1.0/NUM_DATA_HORI, 1.0, 2.0 / NUM_DATA_HORI)
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
                if self.neighbor_topo[i].sources[j] == self.rank: self.from_neighbor[i] = j


        self.data  = self.all_data[self.rank]
        self.neighbor_data = [self.all_data[x] for x in self.to_neighbor]


    def test1(self):
        south = (self.data[SOUTH], self.data[SOUTH][::-1])
        north = (self.data[NORTH], self.data[NORTH][::-1])
        west  = (self.data[WEST], self.data[WEST][::-1])
        east  = (self.data[EAST], self.data[EAST][::-1])
        request = self.topo.start_exchange_vectors(south, north, west, east, self.coord, self.coord)
        result = [(), (), (), ()]
        result[SOUTH], result[NORTH], result[WEST], result[EAST] = request.wait()

        sys.stdout.flush()

        for dir in [SOUTH, NORTH, WEST, EAST]:
            other0 = self.neighbor_data[dir][self.from_neighbor[dir]]
            other1 = other0[::-1]
            r0_other, r1_other = self.neighbor_topo[dir].convert_contra[self.from_neighbor[dir]](other0, other1, self.coord)
            if self.neighbor_topo[dir].flip[self.from_neighbor[dir]]:
                r0_other = numpy.flip(r0_other)
                r1_other = numpy.flip(r1_other)
            diff_s = numpy.linalg.norm(result[dir][0] - r0_other + result[dir][1] - r1_other)
            self.assertLess(diff_s, 1e-15, f'rank {self.rank}: {dir} data is wrong (norm {diff_s:.2e})\n'
                                        f' expected {r0_other}\n'
                                        f'          {r1_other}\n'
                                        f' got      {result[dir][0]}\n'
                                        f'          {result[dir][1]}')
            

    def test2(self):
        south = self.data[SOUTH]
        north = self.data[NORTH]
        west  = self.data[WEST]
        east  = self.data[EAST]
        request = self.topo.start_exchange_scalars(south, north, west, east)
        result = [None, None, None, None]
        result[SOUTH], result[NORTH], result[WEST], result[EAST] = request.wait()

        for dir in [SOUTH, NORTH, WEST, EAST]:
            other = self.neighbor_data[dir][self.from_neighbor[dir]]
            if self.neighbor_topo[dir].flip[self.from_neighbor[dir]]:
                other = numpy.flip(other)
            diff = numpy.linalg.norm(result[dir] - other)
            self.assertLess(diff, 1e-15, f'rank {self.rank}: {dir} data is wrong (norm {diff:.2e})\n')

