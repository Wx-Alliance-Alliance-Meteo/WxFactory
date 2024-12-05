#!/usr/bin/env python3

import argparse
import glob
import os
import re
import sys
from time import time
from typing import Dict, Optional, Tuple, Union

from mpi4py import MPI
import numpy
from numpy import save, load, real, imag
from numpy.linalg import eigvals
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.sparse import csc_matrix, load_npz
import scipy.sparse.linalg

# We assume the script is in a subfolder of the main project
main_gef_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(main_gef_dir)

from eigenvalue_util import gen_matrix, tqdm, store_matrix_set, load_matrix_set

# from run                     import create_geometry, create_preconditioner
from common.configuration import Configuration
from common.process_topology import ProcessTopology
from geometry import DFROperators
from init.init_state_vars import init_state_vars
from rhs.rhs_selector import RhsBundle
from solvers import MatvecOp, MatvecOpRat, fgmres
from simulation import Simulation


# num_el = 0
# order = 0
# num_var = 3
# num_tile = 0


def permutations_3d(num_tiles: int, num_elem_h: int, num_elem_v: int, order: int, num_vars: int):
    p = []
    for t in range(num_tiles):  # |
        for e1 in range(num_elem_h):  # |
            for e2 in range(num_elem_h):  # |
                for e3 in range(num_elem_v):  # |   <- The order we want
                    for o1 in range(order):  # |
                        for o2 in range(order):  # |
                            for o3 in range(order):  # |
                                for v in range(num_vars):  # |
                                    p.append(
                                        num_vars * num_elem_v * order * num_elem_h * order * num_elem_h * order * t  # |
                                        + num_elem_v * order * num_elem_h * order * num_elem_h * order * v  # |
                                        + order * num_elem_h * order * num_elem_h * order * e3  # |
                                        + num_elem_h * order * num_elem_h * order * o3  # | <- The order we have
                                        + order * num_elem_h * order * e1  # |
                                        + num_elem_h * order * o1  # |
                                        + order * e2  # |
                                        + o2
                                    )  # |
    return p


def get_matvecs(cfg_file: str, state_file: Optional[str] = None, build_imex: bool = False) -> Dict:
    """
    Return the initial condition and function handle to compute the action of the Jacobian on a vector
    :param cfg_file: path to the configuration file
    :param state_file: [optional] Path to a file containing a state vector, to use as the initial state
    :param build_imex: [optional] Whether to generate the jacobian for the implicit/explicit splitting
                       (if such a splitting exists)
    :return: (Q, set of matvec functions)
    """

    # Initialize the problem
    sim = Simulation(cfg_file)
    # param = Configuration(cfg_file, MPI.COMM_WORLD.rank == 0)
    # ptopo = ProcessTopology() if param.grid_type == 'cubed_sphere' else None
    # geom = create_geometry(param, ptopo)
    # mtrx = DFROperators(geom, param)
    # Q, topo, metric = init_state_vars(geom, mtrx, param)
    # rhs = RhsBundle(geom, mtrx, metric, topo, ptopo, param, Q.shape)
    # preconditioner = create_preconditioner(param, ptopo, Q)

    if state_file is not None:
        sim.initial_Q = load(state_file)

    Q = sim.initial_Q

    # Create the matvec function(s)
    matvecs = {}
    matvecs["sim"] = sim
    matvecs["config"] = sim.config
    matvecs["rhs"] = sim.rhs.full(Q)
    if sim.rhs.full is not None:
        matvecs["all"] = MatvecOpRat(sim.config.dt, Q, matvecs["rhs"], sim.rhs.full)
    if hasattr(sim.rhs, "implicit") and build_imex:
        matvecs["imp"] = MatvecOpRat(sim.config.dt, Q, sim.rhs.implicit(Q), sim.rhs.implicit)
    if hasattr(sim.rhs, "explicit") and build_imex:
        matvecs["exp"] = MatvecOpRat(sim.config.dt, Q, sim.rhs.explicit(Q), sim.rhs.explicit)
    if sim.preconditioner is not None:
        sim.preconditioner.prepare(sim.config.dt, Q, None)
        matvecs["precond"] = sim.preconditioner

    return matvecs


def compute_eig_from_file(
    jac_file: str, eig_file_name: Optional[str] = None, max_num_val: int = 0
) -> Optional[numpy.ndarray]:
    """Compute the eigenvalues of the matrix stored in the given file."""
    print(f"Loading {os.path.relpath(jac_file)} (compute eig)")
    J = load_npz(f"{jac_file}")
    return compute_eig(J, eig_file_name, max_num_val)


def compute_eig_from_op(
    matvec: MatvecOp, eig_file_name: Optional[str] = None, max_num_val: int = 0
) -> Optional[numpy.ndarray]:
    """Compute the eigenvalues of the matrix represented by the given operator."""
    J = gen_matrix(matvec)
    if MPI.COMM_WORLD.rank == 0:
        return compute_eig(J, eig_file_name, max_num_val)
    return None


def compute_eig(J: csc_matrix, eig_file_name: Optional[str] = None, max_num_val: int = 0) -> numpy.ndarray:
    """
    Compute and save the eigenvalues of a matrix
    :param J:             [in]  Jacobian matrix from which to compute the eigenvalues
    :param eig_file_name: [out] Path to the file where the eigenvalues will be stored
    :param max_num_val:   [in]  How many eigenvalues to compute (0 means compute alllll values).
                          Limiting the number of values is only useful for very large matrices, and can compute
                          few values in a reasonable time.
    """
    # Determine whether we compute alllll eigenvalues of J
    num_val = J.shape[0]
    if num_val > max_num_val and max_num_val > 0:
        num_val = min(max_num_val, J.shape[0] - 2)
    print(f"Computing {num_val} of {J.shape[0]} eigenvalues")

    t0 = time()
    if J.shape[0] == num_val:
        # Compute every eigenvalue (somewhat fast, but has a size limit)
        eig = eigvals(J.toarray())
    else:
        # Compute k eigenvalues (kinda slow, but can work on very large matrices)
        eig, _ = scipy.sparse.linalg.eigs(J, k=num_val)
    t1 = time()

    print(f"Computed {num_val} eigenvalues in {t1 - t0:.1f} s")
    if eig_file_name is not None:
        print(f"Saving {os.path.relpath(eig_file_name)}")
        save(eig_file_name, eig)

    return eig


def plot_eig_from_file(eig_file_name: str, plot_file: Union[str, PdfPages], normalize: bool = True):
    """Plot the eigenvalues stored in the given file."""
    print(f"Loading {os.path.relpath(eig_file_name)} (plot eig)")
    eig = load(eig_file_name)
    plot_eig(eig, plot_file, normalize)

    return eig


def plot_eig_from_operator(matvec: MatvecOp, plot_file: Union[str, PdfPages], normalize: bool = True):
    """Plot the eigenvalues computed from the given matvec operator"""
    eig = compute_eig_from_op(matvec)
    if MPI.COMM_WORLD.rank == 0:
        plot_eig(eig, plot_file, normalize)

    return eig


def plot_eig(eig: numpy.ndarray, plot_file: Union[str, PdfPages], normalize: bool = True):
    """
    Plot the eigenvalues of a matrix
    :param eig: The eigenvalues to plot
    :param plot_file: Path to the file where the plot will be saved. Can also be a PdfPages to have more than one
                      figure on a single pdf.
    :param normalize: If True then the eigenvalues are normalized such that max |e_i| = 1
    """
    # print(f'Loading {os.path.relpath(eig_file_name)} (plot eig)')
    # eig = load(eig_file_name)
    if normalize:
        eig /= numpy.max(numpy.abs(eig))

    if isinstance(plot_file, str):
        pdf = PdfPages(plot_file)
    elif isinstance(plot_file, PdfPages):
        pdf = plot_file
    else:
        raise Exception("Wrong plot file format")

    plt.figure(figsize=(20, 10))
    plt.plot(real(eig), imag(eig), ".")
    plt.hlines(0, min(real(eig)), numpy.max(real(eig)), "k")
    plt.vlines(0, min(imag(eig)), numpy.max(imag(eig)), "k")
    pdf.savefig(bbox_inches="tight")
    plt.close()


def plot_spy_from_file(jac_file_name: str, plot_file: Union[str, PdfPages], prec: float = 0) -> Optional[csc_matrix]:
    """Plot the sparsity pattern of the matrix stored in the given file."""
    print(f"Loading {os.path.relpath(jac_file_name)} (plot spy)")
    J = load_npz(jac_file_name)
    plot_spy(J, plot_file, prec)

    return J


def plot_spy_from_operator(matvec: MatvecOp, plot_file: Union[str, PdfPages], prec: float = 0) -> Optional[csc_matrix]:
    """Plot the sparsity pattern of the matrix represented by the given operator."""
    J = gen_matrix(matvec)
    if MPI.COMM_WORLD.rank == 0:
        plot_spy(J, plot_file, prec)

    return J


def plot_spy(J, plot_file, prec=0):
    """
    Plot the spy of a matrix
    :param J: Jacobian matrix to plot
    :param plot_file: Path to the file where the plot will be saved. Can also be a PdfPages to have more than one figure on a single pdf.
    :param prec: If precision is 0, any non-zero value will be plotted. Otherwise, values of |Z|>precision will be plotted.
    """

    if isinstance(plot_file, str):
        pdf = PdfPages(plot_file)
    elif isinstance(plot_file, PdfPages):
        pdf = plot_file
    else:
        raise Exception("Wrong plot file format")

    plt.figure(figsize=(20, 20))
    plt.spy(J.toarray(), precision=prec)
    pdf.savefig(bbox_inches="tight")
    plt.close()


def output_dir(name: str):
    return os.path.join(main_gef_dir, "jacobian", name)


def jac_file(name: str, rhs: str):
    return os.path.join(output_dir(name), f"J_{rhs}.npz")


def rhs_file(name: str, rhs: str):
    return os.path.join(output_dir(name), f"rhs_{rhs}.npy")


def pdf_spy_file(name: str):
    return os.path.join(output_dir(name), "spy.pdf")


def pdf_eig_file(name: str):
    return os.path.join(output_dir(name), "eig.pdf")


def main(args):
    name = args.name

    # Make sure the output directory exists
    if MPI.COMM_WORLD.rank == 0:
        os.makedirs(output_dir(name), exist_ok=True)
    MPI.COMM_WORLD.Barrier()

    if args.gen_case is not None:
        config = args.gen_case

        matvecs = get_matvecs(config, state_file=args.from_state_file, build_imex=args.imex)
        matrix_set = {}

        param = matvecs["config"]
        num_vars = 4
        if param.equations == "euler" and param.grid_type == "cubed_sphere":
            num_vars = 5
        p = permutations_3d(
            MPI.COMM_WORLD.size, param.nb_elements_horizontal, param.nb_elements_vertical, param.num_solpts, num_vars
        )

        for rhs_name in ["all", "implicit", "explicit"]:
            if rhs_name not in matvecs:
                continue
            matvec = matvecs[rhs_name]
            device = matvecs["sim"].device
            mat = gen_matrix(matvec, jac_file_name=jac_file(name, rhs_name), device=device)
            initial_rhs = MPI.COMM_WORLD.gather(numpy.ravel(matvecs["rhs"]))

            if mat is not None:
                print(f"mat size: {mat.shape}, permutations size: {len(p)}")
                permuted_mat = csc_matrix((mat[p, :])[:, p])
                print(f", permuted shape {permuted_mat.shape}")

                scipy.sparse.save_npz(jac_file(name, rhs_name + ".p"), permuted_mat)
                initial_rhs = numpy.hstack(initial_rhs)

                # print(f'initial rhs = \n{initial_rhs}')
                with open(rhs_file(name, rhs_name), "wb") as f:
                    numpy.save(f, initial_rhs)
                with open(rhs_file(name, rhs_name + ".p"), "wb") as f:
                    numpy.save(f, initial_rhs[p])

                matrix_set[rhs_name] = {}
                matrix_set[rhs_name]["rhs"] = initial_rhs
                matrix_set[rhs_name]["rhs_p"] = initial_rhs[p]
                matrix_set[rhs_name]["matrix"] = mat
                matrix_set[rhs_name]["matrix_p"] = permuted_mat

        if "precond" in matvecs:
            mat = gen_matrix(matvecs["precond"])
            if mat is not None:
                permuted_mat = csc_matrix((mat[p, :])[:, p])
                matrix_set["precond"] = mat
                matrix_set["precond_p"] = permuted_mat

        if len(matrix_set) > 0:
            matrix_set["param"] = param
            matrix_set["permutation"] = p
            store_matrix_set(jac_file(name, ""), matrix_set)

        if MPI.COMM_WORLD.rank == 0:
            print(f"Generation done")

    if (args.plot_all or args.plot_eig or args.plot_spy) and MPI.COMM_WORLD.rank == 0:

        j_files = sorted(glob.glob(output_dir(name) + "/J_*"))

        if len(j_files) > 0:
            if args.plot_all or args.plot_eig:
                pdf_eig = PdfPages(pdf_eig_file(name))
                j_file_pattern = re.compile(r"/J(_.+)\.npz")
                for j_file in j_files:
                    e_file = j_file_pattern.sub(r"/eig\1.npy", j_file)
                    compute_eig_from_file(j_file, e_file, max_num_val=args.max_num_eigval)
                    plot_eig_from_file(e_file, pdf_eig)
                pdf_eig.close()

            if args.plot_all or args.plot_spy:
                pdf_spy = PdfPages(pdf_spy_file(name))
                for j_file in j_files:
                    plot_spy_from_file(j_file, pdf_spy)
                pdf_spy.close()

        else:
            print(
                f'There does not seem to be a generated matrix for problem "{name}".'
                f' Please generate one with the "--gen-case" option.'
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""Generate or plot system matrix and eigenvalues""")
    parser.add_argument("--gen-case", type=str, default=None, help="Generate a system matrix from given case file")
    parser.add_argument("--plot-all", action="store_true", help="Plot the given system matrix and its eigenvalues")
    parser.add_argument("--plot-eig", action="store_true", help="Plot the given system matrix eigenvalues")
    parser.add_argument("--plot-spy", action="store_true", help="Plot the given system matrix")
    parser.add_argument(
        "--from-state-file",
        type=str,
        default=None,
        help="Generate system matrix from a given state vector. (Still have to specify a config file)",
    )
    parser.add_argument(
        "--max-num-eigval", type=int, default=0, help="Maximum number of eigenvalues to compute (0 means all of them)"
    )
    parser.add_argument(
        "--permute",
        action="store_true",
        help="Permute the jacobian matrix so that all nodes/variables" " associated with an element form a block",
    )
    parser.add_argument(
        "--imex",
        action="store_true",
        default=False,
        help="Also construct the matrix/plot for the implicit/explicit RHS's, if available",
    )
    parser.add_argument("name", type=str, help="Name of the case/system matrix")

    main(parser.parse_args())
