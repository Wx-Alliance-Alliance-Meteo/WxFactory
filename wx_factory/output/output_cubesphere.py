import math
from typing import List, Optional

from mpi4py import MPI
import numpy
from numpy.typing import NDArray

from common import Configuration
from device import Device
from geometry import CubedSphere, CubedSphere3D, DFROperators, Metric2D, Metric3DTopo
from init.shallow_water_test import height_vortex, height_case1, height_case2, height_unsteady_zonal
from wx_mpi import ProcessTopology

from .diagnostic import total_energy, potential_enstrophy, global_integral_2d
from .output_manager import OutputManager


class OutputCubesphere(OutputManager):
    def __init__(
        self,
        config: Configuration,
        geometry: CubedSphere,
        operators: DFROperators,
        device: Device,
        metric: Metric2D | Metric3DTopo,
        topography,
        process_topology: ProcessTopology,
    ):
        super().__init__(config, geometry, operators, device)
        self.metric = metric
        self.topo = topography
        self.process_topology = process_topology

        self.rank = self.process_topology.rank
        self.comm = self.process_topology.comm

        self.initial_mass = None
        self.initial_energy = None
        self.initial_enstrophy = None

    def __blockstats__(self, Q, step_id):
        # Blockstats only work for the 2D cubed sphere for now
        if isinstance(self.geometry, CubedSphere3D):
            return

        h = Q[0, :, :]

        if self.config.case_number == 0:
            h_anal, _ = height_vortex(self.geometry, self.metric, self.config, step_id)
        elif self.config.case_number == 1:
            h_anal = height_case1(self.geometry, self.metric, self.config, step_id)
        elif self.config.case_number == 2:
            h_anal = height_case2(self.geometry, self.metric, self.config)
        elif self.config.case_number == 10:
            h_anal = height_unsteady_zonal(self.geometry, self.metric, self.config)

        if self.config.case_number > 1:
            u1_contra = Q[1, :, :] / h
            u2_contra = Q[2, :, :] / h

            if self.config.case_number >= 2:
                energy = total_energy(h, u1_contra, u2_contra, self.topo, self.metric)
                enstrophy = potential_enstrophy(
                    h, u1_contra, u2_contra, self.geometry, self.metric, self.operators, self.config
                )

        if self.rank == 0:
            print("\n================================================================================================")

        if self.config.case_number >= 2:

            if self.initial_mass is None:
                self.initial_mass = global_integral_2d(h, self.operators, self.metric, self.config.num_solpts)
                self.initial_energy = global_integral_2d(energy, self.operators, self.metric, self.config.num_solpts)
                self.initial_enstrophy = global_integral_2d(
                    enstrophy, self.operators, self.metric, self.config.num_solpts
                )

                if self.rank == 0:
                    print(f"Integral of mass = {self.initial_mass}")
                    print(f"Integral of energy = {self.initial_energy}")
                    print(f"Integral of enstrophy = {self.initial_enstrophy}")

        if self.rank == 0:
            print(f"Blockstats for timestep {step_id}")

        if self.config.case_number <= 2 or self.config.case_number == 10:
            absol_err = global_integral_2d(abs(h - h_anal), self.operators, self.metric, self.config.num_solpts)
            int_h_anal = global_integral_2d(abs(h_anal), self.operators, self.metric, self.config.num_solpts)

            absol_err2 = global_integral_2d((h - h_anal) ** 2, self.operators, self.metric, self.config.num_solpts)
            int_h_anal2 = global_integral_2d(h_anal**2, self.operators, self.metric, self.config.num_solpts)

            max_absol_err = self.comm.allreduce(numpy.max(abs(h - h_anal)), op=MPI.MAX)
            max_h_anal = self.comm.allreduce(numpy.max(h_anal), op=MPI.MAX)

            l1 = absol_err / int_h_anal
            l2 = math.sqrt(absol_err2 / int_h_anal2)
            linf = max_absol_err / max_h_anal
            if self.rank == 0:
                print(f"l1 = {l1} \t l2 = {l2} \t linf = {linf}")

        if self.config.case_number >= 2:
            int_mass = global_integral_2d(h, self.operators, self.metric, self.config.num_solpts)
            int_energy = global_integral_2d(energy, self.operators, self.metric, self.config.num_solpts)
            int_enstrophy = global_integral_2d(enstrophy, self.operators, self.metric, self.config.num_solpts)

            normalized_mass = (int_mass - self.initial_mass) / self.initial_mass
            normalized_energy = (int_energy - self.initial_energy) / self.initial_energy
            normalized_enstrophy = (int_enstrophy - self.initial_enstrophy) / self.initial_enstrophy
            if self.rank == 0:
                print(f"normalized error for mass = {normalized_mass}")
                print(f"normalized error for energy = {normalized_energy}")
                print(f"normalized error for enstrophy = {normalized_enstrophy}")

        if self.rank == 0:
            print("================================================================================================")

    def _gather_panel(self, field: NDArray) -> Optional[NDArray]:
        """ """
        panel_comm = self.process_topology.panel_comm
        xp = self.device.xp

        if panel_comm.size == 1:
            return field

        if field.ndim not in [1, 2, 3]:
            raise ValueError(f"shape = {field.shape}, but can only deal with 1D, 2D or 3D blocks of data")

        panel_fields = panel_comm.gather(field, root=0)

        if panel_fields is None:  # non-root PEs
            return None

        side = self.process_topology.num_lines_per_panel
        if field.ndim == 1:
            panel_field = xp.concatenate(panel_fields[:side])
        else:
            panel_field = xp.concatenate(
                [xp.concatenate(panel_fields[i * side : (i + 1) * side], axis=-1) for i in range(side)], axis=-2
            )

        return panel_field

    def _gather_field(self, field: NDArray, num_dim: int):
        return self.process_topology.gather_cube(field, num_dim)
