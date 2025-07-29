#!/usr/bin/env python3

import sys
import os

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
src_dir = os.path.join(root_dir, "wx_factory")
sys.path.append(root_dir)
sys.path.append(src_dir)

from mpi4py import MPI
import output
import process_topology
import device
import geometry
import numpy
import output.output_manager

d = device.CpuDevice(MPI.COMM_WORLD)

config, T0 = output.InputManager.read_config_from_save_file('./tests/data/temp/state_vector_47170bb8616a.00000125.npy', MPI.COMM_WORLD)
config.output_dir = "./tests/data/temp/old"
process_topo = process_topology.ProcessTopology(d, MPI.COMM_WORLD.rank)
print((0,) if T0 is None else T0.shape, process_topo.distribute_cube(T0, 4).shape)
T0 = process_topo.distribute_cube(T0, 4)


total_num_elements_horizontal = config.num_elements_horizontal
num_pe_per_tile = MPI.COMM_WORLD.size // 6
num_pe_per_line = int(numpy.sqrt(num_pe_per_tile))
num_elements_horizontal = total_num_elements_horizontal // num_pe_per_line
cs = geometry.CubedSphere3D(num_elements_horizontal, config.num_elements_vertical, config.num_solpts, total_num_elements_horizontal,
                             config.lambda0, config.phi0, config.alpha0, config.ztop, process_topo, config)


shape = (5,) + (config.num_elements_vertical * config.num_solpts,) + (num_elements_horizontal * config.num_solpts,) * 2
T0_old = numpy.empty(shape)

for i in range(5):
    T0_old[i, ...] = cs.to_single_block(T0[i])

print(T0_old.shape)

om = output.output_manager.OutputManager(config, cs, None, d)
om.step(process_topo.gather_cube(T0_old, 4), 125)