import math

from mpi4py import MPI
import numpy

from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta
from geometry import Cartesian2D
from init.shallow_water_test import height_vortex, height_case1, height_case2, height_unsteady_zonal
from output.diagnostic import total_energy, potential_enstrophy, global_integral


def blockstats_cs(Q, geom, topo, metric, mtrx, param, step):

    h = Q[0, :, :]

    if param.case_number == 0:
        h_anal, _ = height_vortex(geom, metric, param, step)
    elif param.case_number == 1:
        h_anal = height_case1(geom, metric, param, step)
    elif param.case_number == 2:
        h_anal = height_case2(geom, metric, param)
    elif param.case_number == 10:
        h_anal = height_unsteady_zonal(geom, metric, param)

    if param.case_number > 1:
        u1_contra = Q[1, :, :] / h
        u2_contra = Q[2, :, :] / h

    if param.case_number >= 2:
        energy = total_energy(h, u1_contra, u2_contra, geom, topo, metric)
        enstrophy = potential_enstrophy(h, u1_contra, u2_contra, geom, metric, mtrx, param)

    if MPI.COMM_WORLD.rank == 0:
        print("\n================================================================================================")

    if param.case_number >= 2:
        global initial_mass
        global initial_energy
        global initial_enstrophy

        try:
            if initial_mass is None and MPI.COMM_WORLD.rank == 0:
                print("Nothing!!!!!")
        except NameError:
            initial_mass = global_integral(h, mtrx, metric, param.nbsolpts, param.nb_elements_horizontal)
            initial_energy = global_integral(energy, mtrx, metric, param.nbsolpts, param.nb_elements_horizontal)
            initial_enstrophy = global_integral(enstrophy, mtrx, metric, param.nbsolpts, param.nb_elements_horizontal)

            if MPI.COMM_WORLD.rank == 0:
                print(f"Integral of mass = {initial_mass}")
                print(f"Integral of energy = {initial_energy}")
                print(f"Integral of enstrophy = {initial_enstrophy}")

    if MPI.COMM_WORLD.rank == 0:
        print("Blockstats for timestep ", step)

    if param.case_number <= 2 or param.case_number == 10:
        absol_err = global_integral(abs(h - h_anal), mtrx, metric, param.nbsolpts, param.nb_elements_horizontal)
        int_h_anal = global_integral(abs(h_anal), mtrx, metric, param.nbsolpts, param.nb_elements_horizontal)

        absol_err2 = global_integral((h - h_anal) ** 2, mtrx, metric, param.nbsolpts, param.nb_elements_horizontal)
        int_h_anal2 = global_integral(h_anal**2, mtrx, metric, param.nbsolpts, param.nb_elements_horizontal)

        max_absol_err = MPI.COMM_WORLD.allreduce(numpy.max(abs(h - h_anal)), op=MPI.MAX)
        max_h_anal = MPI.COMM_WORLD.allreduce(numpy.max(h_anal), op=MPI.MAX)

        l1 = absol_err / int_h_anal
        l2 = math.sqrt(absol_err2 / int_h_anal2)
        linf = max_absol_err / max_h_anal
        if MPI.COMM_WORLD.rank == 0:
            print(f"l1 = {l1} \t l2 = {l2} \t linf = {linf}")

    if param.case_number >= 2:
        int_mass = global_integral(h, mtrx, metric, param.nbsolpts, param.nb_elements_horizontal)
        int_energy = global_integral(energy, mtrx, metric, param.nbsolpts, param.nb_elements_horizontal)
        int_enstrophy = global_integral(enstrophy, mtrx, metric, param.nbsolpts, param.nb_elements_horizontal)

        normalized_mass = (int_mass - initial_mass) / initial_mass
        normalized_energy = (int_energy - initial_energy) / initial_energy
        normalized_enstrophy = (int_enstrophy - initial_enstrophy) / initial_enstrophy
        if MPI.COMM_WORLD.rank == 0:
            print(f"normalized error for mass = {normalized_mass}")
            print(f"normalized error for energy = {normalized_energy}")
            print(f"normalized error for enstrophy = {normalized_enstrophy}")

    if MPI.COMM_WORLD.rank == 0:
        print("================================================================================================")


def blockstats_cart(Q: numpy.ndarray, geom: Cartesian2D, step_id: int):

    def get_stats(field, geom):
        f_minloc = numpy.unravel_index(field.argmin(), field.shape)
        f_maxloc = numpy.unravel_index(field.argmax(), field.shape)
        f_mincoord = (geom.X1[f_minloc], geom.X3[f_minloc])
        f_maxcoord = (geom.X1[f_maxloc], geom.X3[f_maxloc])
        f_min = field[f_minloc]
        f_max = field[f_maxloc]
        f_avg = field.mean()

        return f_mincoord, f_maxcoord, f_min, f_max, f_avg

    rho = Q[idx_2d_rho]
    rho_mincoord, rho_maxcoord, rho_min, rho_max, rho_avg = get_stats(rho, geom)

    u = Q[idx_2d_rho_u] / rho
    u_mincoord, u_maxcoord, u_min, u_max, u_avg = get_stats(u, geom)

    w = Q[idx_2d_rho_w] / rho
    w_mincoord, w_maxcoord, w_min, w_max, w_avg = get_stats(w, geom)

    theta = Q[idx_2d_rho_theta] / rho
    theta_mincoord, theta_maxcoord, theta_min, theta_max, theta_avg = get_stats(theta, geom)

    if MPI.COMM_WORLD.rank == 0:
        print("==============================================")
        print(f" Blockstats for timestep {step_id}")
        print(f"   Var        Min        Max        Mean")
        print(
            f"  ρ        {rho_min:9.2e}  {rho_max:9.2e}  {rho_avg:9.2e}   "
            # f'({rho_mincoord[0]:7.0f}, {rho_mincoord[1]:7.0f})  '
            # f'({rho_maxcoord[0]:7.0f}, {rho_maxcoord[1]:7.0f})'
        )
        print(
            f"  u        {u_min:9.2e}  {u_max:9.2e}  {u_avg:9.2e}   "
            # f'({u_mincoord[0]:7.0f}, {u_mincoord[1]:7.0f})  '
            # f'({u_maxcoord[0]:7.0f}, {u_maxcoord[1]:7.0f})'
        )
        print(
            f"  w        {w_min:9.2e}  {w_max:9.2e}  {w_avg:9.2e}   "
            # f'({w_mincoord[0]:7.0f}, {w_mincoord[1]:7.0f})  '
            # f'({w_maxcoord[0]:7.0f}, {w_maxcoord[1]:7.0f})'
        )
        print(
            f"  θ        {theta_min:9.2e}  {theta_max:9.2e}  {theta_avg:9.2e}   "
            # f'({theta_mincoord[0]:7.0f}, {theta_mincoord[1]:7.0f})  '
            # f'({theta_maxcoord[0]:7.0f}, {theta_maxcoord[1]:7.0f})'
        )
        print("==============================================")
