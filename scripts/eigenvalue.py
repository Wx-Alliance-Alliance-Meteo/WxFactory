#!/usr/bin/env python3

import argparse
import glob
import os
import re
import sys
from   time      import time
from   typing    import Callable, Dict, Optional, Tuple, Union

from   mpi4py              import MPI
import numpy
from   numpy               import zeros_like, save, load, real, imag, zeros
from   numpy.linalg        import eigvals
import matplotlib.pyplot as plt
from   matplotlib.backends.backend_pdf import PdfPages
from   scipy.sparse        import csc_matrix, save_npz, load_npz, vstack
import scipy.sparse.linalg

# We assume the script is in a subfolder of the main project
main_gef_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(main_gef_dir)

from main_gef import create_geometry, create_preconditioner

from common.program_options import Configuration
from common.parallel        import DistributedWorld
from eigenvalue_util        import gen_matrix, tqdm
from geometry               import DFROperators
from init.init_state_vars   import init_state_vars
from rhs.rhs_selector       import RhsBundle
from solvers                import MatvecOp, MatvecOpRat


# num_el = 0
# order = 0
# num_var = 3
# num_tile = 0

# def permutations():
#    p = []
#    for t in range(num_tile):
#       for e1 in range(num_el):
#          for e2 in range(num_el):
#             for o1 in range(order):
#                for o2 in range(order):
#                   for v in range(num_var):
#                      p.append(num_var*order*order*num_el*num_el*t +
#                            order*order*num_el*num_el*v +
#                            order*order*num_el*e1 +
#                            order*num_el*o1 +
#                            order*e2 + o2)
#    return p

def get_matvecs(cfg_file: str, state_file: Optional[str] = None, build_imex: bool = False) \
      -> Dict[str, MatvecOp]:
   """
   Return the initial condition and function handle to compute the action of the Jacobian on a vector
   :param cfg_file: path to the configuration file
   :param state_file: [optional] Path to a file containing a state vector, to use as the initial state
   :param build_imex: [optional] Whether to generate the jacobian for the implicit/explicit splitting
                      (if such a splitting exists)
   :return: (Q, set of matvec functions)
   """

   # Initialize the problem
   param = Configuration(cfg_file, MPI.COMM_WORLD.rank == 0)
   ptopo = DistributedWorld() if param.grid_type == 'cubed_sphere' else None
   geom = create_geometry(param, ptopo)
   mtrx = DFROperators(geom, param.filter_apply, param.filter_order, param.filter_cutoff)
   Q, topo, metric = init_state_vars(geom, mtrx, param)
   rhs = RhsBundle(geom, mtrx, metric, topo, ptopo, param)
   preconditioner = create_preconditioner(param, ptopo)

   if state_file is not None: Q = load(state_file)

   # Create the matvec function(s)
   matvecs = {}
   if rhs.full is not None:      matvecs['all'] = MatvecOpRat(param.dt, Q, rhs.full(Q), rhs.full)
   if rhs.implicit and build_imex: matvecs['imp'] = MatvecOpRat(param.dt, Q, rhs.implicit(Q), rhs.implicit)
   if rhs.explicit and build_imex: matvecs['exp'] = MatvecOpRat(param.dt, Q, rhs.explicit(Q), rhs.explicit)
   if preconditioner is not None:
      preconditioner.prepare(param.dt, Q, None)
      matvecs['precond'] = preconditioner

   return matvecs

def compute_eig_from_file(jac_file: str, eig_file_name: Optional[str] = None, max_num_val: int = 0) \
      -> Optional[numpy.ndarray]:
   """Compute the eigenvalues of the matrix stored in the given file."""
   print(f'Loading {os.path.relpath(jac_file)} (compute eig)')
   J = load_npz(f'{jac_file}')
   return compute_eig(J, eig_file_name, max_num_val)

def compute_eig_from_op(matvec: MatvecOp, eig_file_name: Optional[str] = None, max_num_val: int = 0) \
      -> Optional[numpy.ndarray]:
   """Compute the eigenvalues of the matrix represented by the given operator."""
   J = gen_matrix(matvec)
   if MPI.COMM_WORLD.rank == 0:
      return compute_eig(J, eig_file_name, max_num_val)
   return None

def compute_eig(J: csc_matrix,
                eig_file_name: Optional[str] = None,
                max_num_val: int = 0) -> numpy.ndarray:
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
   print(f'Computing {num_val} of {J.shape[0]} eigenvalues')

   t0 = time()
   if J.shape[0] == num_val:
      # Compute every eigenvalue (somewhat fast, but has a size limit)
      eig = eigvals(J.toarray())
   else:
      # Compute k eigenvalues (kinda slow, but can work on very large matrices)
      eig, _ = scipy.sparse.linalg.eigs(J, k=num_val)
   t1 = time()

   print(f'Computed {num_val} eigenvalues in {t1 - t0:.1f} s')
   if eig_file_name is not None:
      print(f'Saving {os.path.relpath(eig_file_name)}')
      save(eig_file_name, eig)

   return eig

def plot_eig_from_file(eig_file_name: str, plot_file: Union[str, PdfPages], normalize: bool = True):
   """Plot the eigenvalues stored in the given file."""
   print(f'Loading {os.path.relpath(eig_file_name)} (plot eig)')
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
      raise Exception('Wrong plot file format')

   plt.figure(figsize=(20, 10))
   plt.plot(real(eig), imag(eig), '.')
   plt.hlines(0, min(real(eig)), numpy.max(real(eig)), 'k')
   plt.vlines(0, min(imag(eig)), numpy.max(imag(eig)), 'k')
   pdf.savefig(bbox_inches='tight')
   plt.close()

def plot_spy_from_file(jac_file_name: str, plot_file: Union[str, PdfPages], prec: float = 0) -> Optional[csc_matrix]:
   """Plot the sparsity pattern of the matrix stored in the given file."""
   print(f'Loading {os.path.relpath(jac_file_name)} (plot spy)')
   J = load_npz(jac_file_name)
   plot_spy(J, plot_file, prec)

   return J

def plot_spy_from_operator(matvec: MatvecOp, plot_file: Union[str, PdfPages], prec: float = 0) -> Optional[csc_matrix]:
   """Plot the sparsity pattern of the matrix represented by the given operator."""
   J = gen_matrix(matvec)
   if MPI.COMM_WORLD.rank == 0:
      plot_spy(J, plot_file, prec)

   return J

def plot_spy(J, plot_file, prec = 0):
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
      raise Exception('Wrong plot file format')

   plt.figure(figsize=(20, 20))
   plt.spy(J.toarray(), precision=prec)
   pdf.savefig(bbox_inches='tight')
   plt.close()

def output_dir(name):
   return os.path.join(main_gef_dir, 'jacobian', name)

def jac_file(name, rhs):
   return os.path.join(output_dir(name), f'J_{rhs}.npz')

def pdf_spy_file(name):
   return os.path.join(output_dir(name), 'spy.pdf')

def pdf_eig_file(name):
   return os.path.join(output_dir(name), 'eig.pdf')

def main(args):
   name = args.name

   # Make sure the output directory exists
   if MPI.COMM_WORLD.rank == 0: os.makedirs(output_dir(name), exist_ok=True)
   MPI.COMM_WORLD.Barrier()

   if args.gen_case is not None:
      config = args.gen_case

      matvecs = get_matvecs(config, state_file=args.from_state_file, build_imex=args.imex)
      for rhs, matvec in matvecs.items():
         gen_matrix(matvec, jac_file_name=jac_file(name, rhs), permute=args.permute)

      print(f'Generation done')

   if args.plot and MPI.COMM_WORLD.rank == 0:
      pdf_spy = PdfPages(pdf_spy_file(name))
      pdf_eig = PdfPages(pdf_eig_file(name))

      j_files = sorted(glob.glob(output_dir(name) + '/J_*'))

      if len(j_files) == 0:
         print(f'There does not seem to be a generated matrix for problem "{name}".'
               f' Please generate one with the "--gen-case" option.')
      j_file_pattern = re.compile(r'/J(_.+)\.npz')
      for j_file in j_files:
         e_file = j_file_pattern.sub(r'/eig\1.npy', j_file)
         compute_eig_from_file(j_file, e_file, max_num_val=args.max_num_eigval)
         plot_eig_from_file(e_file, pdf_eig)
         plot_spy_from_file(j_file, pdf_spy)

      pdf_spy.close()
      pdf_eig.close()

if __name__ == '__main__':

   parser = argparse.ArgumentParser(description='''Generate or plot system matrix and eigenvalues''')
   parser.add_argument('--gen-case', type=str, default=None, help='Generate a system matrix from given case file')
   parser.add_argument('--plot', action='store_true', help='Plot the given system matrix and its eigenvalues')
   parser.add_argument('--from-state-file', type=str, default=None,
                       help = 'Generate system matrix from a given state vector. (Still have to specify a config file)')
   parser.add_argument('--max-num-eigval', type=int, default=0,
                       help = 'Maximum number of eigenvalues to compute (0 means all of them)')
   parser.add_argument('--permute', action='store_true', help='Permute the jacobian matrix so that all nodes/variables'
                       ' associated with an element form a block')
   parser.add_argument('--imex', action='store_true', default=False,
                       help="Also construct the matrix/plot for the implicit/explicit RHS's, if available")
   parser.add_argument('name', type=str, help='Name of the case/system matrix')

   main(parser.parse_args())
