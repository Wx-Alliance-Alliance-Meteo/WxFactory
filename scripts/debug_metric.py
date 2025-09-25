#!/usr/bin/env python3

import argparse
import copy
import os
import sys

import cupy
import gmpy2
from mpi4py import MPI
import numpy

src_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "wx_factory")
sys.path.append(src_dir)

from wx_mpi import SingleProcess, Conditional
from device import CpuDevice, CudaDevice
from simulation import Simulation
from output.input_manager import InputManager

cpu_dev = CpuDevice(MPI.COMM_WORLD)
cuda_dev = CudaDevice(MPI.COMM_WORLD)
rank = MPI.COMM_WORLD.rank
numpy.set_printoptions(precision=20, linewidth=130)


def mpfr_norm(a):
    return float(gmpy2.sqrt(numpy.sum(a * a)))


def rel_diff(a, b):
    a_h = a if not isinstance(a, cupy.ndarray) else cuda_dev.to_host(a)
    b_h = b if not isinstance(b, cupy.ndarray) else cuda_dev.to_host(b)
    diff = b_h - a_h
    norm_a = mpfr_norm(a_h)
    if norm_a > 0:
        diff /= norm_a

    diff_norm = mpfr_norm(diff)

    all_diffs = numpy.zeros(a_h.shape[0:2], dtype=numpy.float64)
    a_norms = numpy.zeros_like(all_diffs)
    b_norms = numpy.zeros_like(all_diffs)
    for i in range(all_diffs.shape[0]):
        for j in range(all_diffs.shape[1]):
            a_norms[i, j] = mpfr_norm(a_h[i, j])
            b_norms[i, j] = mpfr_norm(b_h[i, j])
            tmp_diff = diff[i, j].copy()
            if a_norms[i, j] > 0.0:
                tmp_diff /= a_norms[i, j]
            all_diffs[i, j] = mpfr_norm(tmp_diff)

    # if rank == 0:
    #     print(f"shape = {a.shape}", flush=True)
    #     print(f"a norm = {norm_a:.2e}", flush=True)
    #     print(f"a norms = \n{a_norms}", flush=True)
    #     print(f"b norms = \n{b_norms}", flush=True)
    #     print(f"all diffs = \n{all_diffs}", flush=True)
    #     # print(f"a sample: \n{a_h}", flush=True)

    return diff, diff_norm, diff.max(), norm_a, all_diffs  # , diffs, diffs_norm


def main(args):

    with SingleProcess() as s, Conditional(s):
        if (args.config == "") == (args.state_vector == ""):
            print(f"config = {args.config}, state vector = {args.state_vector}", flush=True)
            print(f"Error, must give either a config file or a state vector", flush=True)
            raise SystemExit(1)

    if args.config != "":
        config = InputManager.read_config(args.config, MPI.COMM_WORLD)
    else:
        config, _ = InputManager.read_config_from_save_file(args.state_vector, MPI.COMM_WORLD)

    config.desired_device = "numpy"
    config2 = copy.deepcopy(config)
    config2.desired_device = "cupy"

    # with SingleProcess() as s, Conditional(s):
    #     print(f"Config: {config}", flush=True)

    try:
        sim_cpu = Simulation(config, comm=MPI.COMM_WORLD, quiet=True)
        sim_gpu = Simulation(config2, comm=MPI.COMM_WORLD, quiet=True)

        def print_diff(a, b, verbose):
            diff, diff_norm, diff_max, norm_a, all_diffs = rel_diff(a, b)

            if rank == 0:
                if verbose:
                    print(f"A: \n{a}\nB:\n{b}\ndiff:\n{diff}", flush=True)
                print(f"diff norm {diff_norm:.2e}, max diff {diff_max:.2e}, norm_a {norm_a:.2e}", flush=True)

        print_diff(sim_cpu.metric.height_int, sim_gpu.metric.height_int, verbose=False)
        print_diff(sim_cpu.metric.height_itf_j, sim_gpu.metric.height_itf_j, verbose=True)
        print_diff(sim_cpu.metric.dRdx2, sim_gpu.metric.dRdx2, verbose=True)
        # print_diff(sim_cpu.metric.dRdx1_new, sim_gpu.metric.dRdx1_new, verbose=False)
        # print_diff(sim_cpu.metric.dRdeta_new, sim_gpu.metric.dRdeta_new, verbose=False)
        print_diff(sim_cpu.metric.drx2_a, sim_gpu.metric.drx2_a, verbose=False)
        print_diff(sim_cpu.metric.drx2_b, sim_gpu.metric.drx2_b, verbose=False)
        print_diff(sim_cpu.metric.dRdx2_new, sim_gpu.metric.dRdx2_new, verbose=True)
        print_diff(sim_cpu.operators.correction_SN, sim_gpu.operators.correction_SN, verbose=True)

        # print_diff(sim_cpu.metric.h_contra_new[0, 2], sim_gpu.metric.h_contra_new[0, 2], verbose=False)
        # print_diff(sim_cpu.metric.sqrtG_new, sim_gpu.metric.sqrtG_new, verbose=False)

    except (
        Exception,
        KeyboardInterrupt,
        SystemExit,
    ) as e:  # KeyboardInterrupt does not seem to be caught with just "Exception"...
        if rank == 0:
            raise e
        raise SystemExit(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--state-vector", type=str, default="")

    args = parser.parse_args()

    main(args)
