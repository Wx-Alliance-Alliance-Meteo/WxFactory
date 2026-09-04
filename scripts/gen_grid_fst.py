#!/usr/bin/env python3

import math

import numpy
import torch
from mpi4py import MPI
from rmn import FstDataType, fst24_file, fst_record

from wx_factory.context import Context
from wx_factory.geometry.registry import CubedSphere3D, GeometryContext, resolve_geometry
from wx_factory.output import InputManager
from wx_factory.process_topology import ProcessTopology
from wx_factory.wx_mpi import Conditional, SingleProcess

rank = MPI.COMM_WORLD.Get_rank()


def main(args):
    if rank == 0:
        print(f"Generating grid from config file: {args.config_file}", flush=True)

    context = Context(MPI.COMM_WORLD)
    config = InputManager.read_config(args.config_file, context.comm)
    grid = resolve_geometry(
        GeometryContext(
            config=config,
            context=context,
            comm=context.comm,
            num_elements_horizontal=config.num_elements_horizontal,
            num_solpts=config.num_solpts,
            total_num_elements_horizontal=config.num_elements_horizontal,
            lambda0=config.lambda0,
            phi0=config.phi0,
            alpha0=config.alpha0,
        )
    )

    if not isinstance(grid, CubedSphere3D):
        with SingleProcess() as s, Conditional(s):
            raise TypeError("Can only output cubed-sphere grid at the moment")

    ptopo: ProcessTopology = grid.process_topology

    all_lats_t = ptopo.gather_cube(grid.block_lat * 180.0 / math.pi, 2)
    all_lons_t = ptopo.gather_cube(grid.block_lon * 180.0 / math.pi, 2)

    all_lats = all_lats_t.to(torch.float32).cpu().numpy() if all_lats_t is not None else None
    all_lons = all_lons_t.to(torch.float32).cpu().numpy() if all_lons_t is not None else None

    def make_rec(data, ip=0, name="", etiket=""):
        return fst_record(
            data_bits=32,
            pack_bits=32,
            data_type=FstDataType.FST_TYPE_REAL,
            data=data,
            dateo=0,
            datev=0,
            deet=0,
            npas=0,
            ni=data.size,
            nj=1,
            nk=1,
            ip1=ip,
            ip2=ip,
            ip3=ip,
            ig1=900,
            ig2=0,
            ig3=14400,
            ig4=0,
            nomvar=name[:4],
            etiket=etiket[:12],
            typvar="X",
            grtyp="E",
        )

    split_name = args.output_file.strip().split(".")
    base_name = split_name[0]
    suffix = "" if len(split_name) < 2 else split_name[1]

    filenames = [f"{base_name}_{i}" + f".{suffix}" if suffix != "" else "" for i in range(6)]

    with SingleProcess() as s, Conditional(s):
        if all_lats is None or all_lons is None:
            raise ValueError("Process does not have lon/lat")
        for i in range(6):
            with fst24_file(filenames[i], "R/W+XDF") as f:
                f.write(make_rec(all_lons[i].T, ip=i, name=">>", etiket=f"PANEL{i}"), False)
                rec = make_rec(all_lats[i].T, ip=i, name="^^", etiket=f"PANEL{i}")
                rec.nj = rec.ni
                rec.ni = 1
                f.write(rec, False)

    with SingleProcess() as s, Conditional(s), fst24_file(filenames, "R/O") as f:
        numpy.set_printoptions(linewidth=160, precision=3)
        for rec in f.new_query():
            a = all_lons if rec.nomvar == ">>" else all_lats
            # print(
            #     f"Shape = {rec.data.shape}, \n"  # nofmt
            #     f"ref flags: \n{a[rec.ig1].flags}, \n"
            #     f"rec flags: \n{rec.data.flags}",
            #     flush=True,
            # )
            diff = numpy.ravel(a[rec.ip1]) - numpy.ravel(rec.data)
            diff_norm = numpy.linalg.norm(diff)
            if diff_norm > 0.0:
                print(
                    f"Diff {rec.nomvar}/{rec.etiket} = {diff_norm:.2e}\n"  # nofmt
                    f"ref = \n{a[rec.ip1]}\n"
                    f"got \n{rec.data}\n"
                    f"diff: \n{diff}",
                    flush=True,
                )
                raise ValueError("Not the same!")


if __name__ == "__main__":
    if MPI.COMM_WORLD.size != 6:
        if rank == 0:
            raise RuntimeError("This script must be run with 6 MPI ranks.")
        else:
            raise SystemExit(1)

    import argparse

    args = None
    if rank == 0:
        parser = argparse.ArgumentParser(
            description="Generate a standard file containing the provided grid. "
            "That file will be used to generate topography for that grid."
        )
        parser.add_argument(
            "config_file", type=str, help="Path to the configuration file where grid parameters are found."
        )
        parser.add_argument("--output-file", type=str, default="grid.fst", help="File where the grid will be saved")
        args = parser.parse_args()

    args = MPI.COMM_WORLD.bcast(args)
    main(args)
