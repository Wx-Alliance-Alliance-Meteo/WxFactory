import math

from mpi4py import MPI

from common import Configuration
from device import Device
from geometry import CubedSphere, CubedSphere3D, DFROperators, Metric2D, Metric3DTopo
from init.shallow_water_test import height_vortex, height_case1, height_case2, height_unsteady_zonal

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
        topo,
    ):
        super().__init__(config, geometry, operators, device)
        self.metric = metric
        self.topo = topo

        self.initial_mass = None
        self.initial_energy = None
        self.initial_enstrophy = None

    def __blockstats__(self, Q, step_id):
        # Blockstats only work for the 2D cubed sphere for now
        if isinstance(self.geometry, CubedSphere3D):
            return

        h = Q[0, :, :]

        if self.param.case_number == 0:
            h_anal, _ = height_vortex(self.geometry, self.metric, self.param, step_id)
        elif self.param.case_number == 1:
            h_anal = height_case1(self.geometry, self.metric, self.param, step_id)
        elif self.param.case_number == 2:
            h_anal = height_case2(self.geometry, self.metric, self.param)
        elif self.param.case_number == 10:
            h_anal = height_unsteady_zonal(self.geometry, self.metric, self.param)

        if self.param.case_number > 1:
            u1_contra = Q[1, :, :] / h
            u2_contra = Q[2, :, :] / h

            if self.param.case_number >= 2:
                energy = total_energy(h, u1_contra, u2_contra, self.topo, self.metric)
                enstrophy = potential_enstrophy(
                    h, u1_contra, u2_contra, self.geometry, self.metric, self.operators, self.param
                )

        if MPI.COMM_WORLD.rank == 0:
            print("\n================================================================================================")

        if self.param.case_number >= 2:

            if self.initial_mass is None:
                self.initial_mass = global_integral_2d(h, self.operators, self.metric, self.param.num_solpts)
                self.initial_energy = global_integral_2d(energy, self.operators, self.metric, self.param.num_solpts)
                self.initial_enstrophy = global_integral_2d(
                    enstrophy, self.operators, self.metric, self.param.num_solpts
                )

                if MPI.COMM_WORLD.rank == 0:
                    print(f"Integral of mass = {self.initial_mass}")
                    print(f"Integral of energy = {self.initial_energy}")
                    print(f"Integral of enstrophy = {self.initial_enstrophy}")

        if MPI.COMM_WORLD.rank == 0:
            print(f"Blockstats for timestep {step_id}")

        if self.param.case_number <= 2 or self.param.case_number == 10:
            absol_err = global_integral_2d(abs(h - h_anal), self.operators, self.metric, self.param.num_solpts)
            int_h_anal = global_integral_2d(abs(h_anal), self.operators, self.metric, self.param.num_solpts)

            absol_err2 = global_integral_2d((h - h_anal) ** 2, self.operators, self.metric, self.param.num_solpts)
            int_h_anal2 = global_integral_2d(h_anal**2, self.operators, self.metric, self.param.num_solpts)

            max_absol_err = MPI.COMM_WORLD.allreduce(numpy.max(abs(h - h_anal)), op=MPI.MAX)
            max_h_anal = MPI.COMM_WORLD.allreduce(numpy.max(h_anal), op=MPI.MAX)

            l1 = absol_err / int_h_anal
            l2 = math.sqrt(absol_err2 / int_h_anal2)
            linf = max_absol_err / max_h_anal
            if MPI.COMM_WORLD.rank == 0:
                print(f"l1 = {l1} \t l2 = {l2} \t linf = {linf}")

        if self.param.case_number >= 2:
            int_mass = global_integral_2d(h, self.operators, self.metric, self.param.num_solpts)
            int_energy = global_integral_2d(energy, self.operators, self.metric, self.param.num_solpts)
            int_enstrophy = global_integral_2d(enstrophy, self.operators, self.metric, self.param.num_solpts)

            normalized_mass = (int_mass - self.initial_mass) / self.initial_mass
            normalized_energy = (int_energy - self.initial_energy) / self.initial_energy
            normalized_enstrophy = (int_enstrophy - self.initial_enstrophy) / self.initial_enstrophy
            if MPI.COMM_WORLD.rank == 0:
                print(f"normalized error for mass = {normalized_mass}")
                print(f"normalized error for energy = {normalized_energy}")
                print(f"normalized error for enstrophy = {normalized_enstrophy}")

        if MPI.COMM_WORLD.rank == 0:
            print("================================================================================================")
