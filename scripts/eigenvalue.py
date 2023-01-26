#!/usr/bin/env python3

import sys
import os
from   itertools import product
from   time      import time
import argparse

try:
   from tqdm import tqdm
except ModuleNotFoundError:
   print(f'Module "tqdm" was not found. You need it if you want to see progress bars')
   def tqdm(a): return a

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

from main_gef import create_geometry

from Common.program_options import Configuration
from Common.parallel        import Distributed_World
from Geometry.matrices      import DFR_operators
from Init.init_state_vars   import init_state_vars
from Solver.matvec          import matvec_fun, matvec_rat


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

def get_matvec(cfg_file, rhs, state_file = None):
   """
   Return the initial condition and function handle to compute the action of the Jacobian on a vector
   :param cfg_file: path to the configuration file
   :param rhs: type of rhs can be 'all', 'exp' or 'imp'
   :return: (Q, matvec)
   """
   param = Configuration(cfg_file, MPI.COMM_WORLD.rank == 0)
   ptopo = Distributed_World() if param.grid_type == 'cubed_sphere' else None
   geom = create_geometry(param, ptopo)
   mtrx = DFR_operators(geom, param.filter_apply, param.filter_order, param.filter_cutoff)
   Q, _, _, rhs_handle, rhs_implicit, rhs_explicit = init_state_vars(geom, mtrx, ptopo, param)
   if state_file is not None: Q = load(state_file)

   # global num_el, order, num_tile
   # num_el   = param.nb_elements_horizontal
   # order    = param.nbsolpts
   # num_tile = MPI.COMM_WORLD.size

   if rhs == 'all':
      rhs = rhs_handle
   elif rhs == 'exp':
      rhs = rhs_explicit
   elif rhs == 'imp':
      rhs = rhs_implicit
   else:
      raise Exception('Wrong rhs name')

   # return (Q, lambda v: matvec_fun(v, 1, Q, rhs), rhs)
   rhs_vec = rhs(Q)
   return Q, lambda v: matvec_rat(v, param.dt, Q, rhs_vec, rhs), rhs

def gen_matrix(Q, matvec, jac_file_name, permute):
   """
   Compute and store the Jacobian matrix. It may be computed either as a full or sparse matrix
   (faster as full, but it may take a *lot* of memory for large matrices). Always stored as
   sparse.
   :param Q: Solution vector where the Jacobian is computed
   :param matvec: Function handle to compute the action of the jacobian on a vector
   :param jac_file_name: Path to the file where the jacobian will be stored
   :param permute: Whether to permute matrix rows and columns to groups entries associated with an element into a block
   """
   neq, ni, nj = Q.shape
   n_loc = Q.size
   compressed = n_loc > 150000

   rank = MPI.COMM_WORLD.Get_rank()
   size = MPI.COMM_WORLD.Get_size()

   Qid = zeros_like(Q)
   J = csc_matrix((n_loc, size*n_loc), dtype=Q.dtype) if compressed else zeros((n_loc, size*n_loc))

   def progress(a): return a
   if rank == 0:
      progress = tqdm

   idx = 0
   indices = [i for i in product(range(neq), range(ni), range(nj))]
   for r in range(size):
      if rank == 0: print(f'Tile {r+1}/{size}')
      for (i, j, k) in progress(indices):
         if rank == r: Qid[i, j, k] = 1.0
         col = matvec(Qid.flatten())
         J[:, idx] = csc_matrix(col).transpose() if compressed else col
         idx += 1
         Qid[i, j, k] = 0.0

   # If it wasn't already compressed, do it now
   if not compressed: J = csc_matrix(J)

   print(f'gathering')
   J_comm   = MPI.COMM_WORLD.gather(J, root=0)
   print(f'gathering done')

   if rank == 0:
      print('')

      glb_J = vstack(J_comm)

      if permute:
         print(f'Permute was not implemented for cartesian grids and (possibly) 3D problems')
         # p = permutations()
         # glb_J = csc_matrix(lil_matrix(glb_J)[p, :][:, p])

      save_npz(jac_file_name, glb_J)
      print(f'Matrix saved')

def compute_eig(jac_file_name, eig_file_name, max_num_vals):
   """
   Compute and save the eigenvalues of a matrix
   :param jac_file_name: Path to the file where the matrix is stored
   :param eig_file_name: Path to the file where the eigenvalues will be stored
   """
   print(f'Loading {jac_file_name} (compute eig)')
   J = load_npz(f'{jac_file_name}')

   # Determine whether we compute alllll eigenvalues of J
   num_vals = J.shape[0]
   if J.shape[0] > max_num_vals and max_num_vals > 0:
      num_vals = min(max_num_vals, J.shape[0] - 2)
   print(f'Computing {num_vals} of {J.shape[0]} eigenvalues')

   t0 = time()
   if J.shape[0] == num_vals:
      # Compute every eigenvalue (somewhat fast, but has a size limit)
      eig = eigvals(J.toarray())
   else:
      # Compute k eigenvalues (kinda slow, but can work on very large matrices)
      eig, _ = scipy.sparse.linalg.eigs(J, k=num_vals)
   t1 = time()

   print(f'Computed {num_vals} eigenvalues in {t1 - t0:.1f} s')
   print(f'Saving {eig_file_name}')
   save(eig_file_name, eig)


def plot_eig(eig_file_name, plot_file, normalize=True):
   """
   Plot the eigenvalues of a matrix
   :param eig_file_name: Path to the file where the eigenvalues are stored
   :param plot_file: Path to the file where the plot will be saved. Can also be a PdfPages to have more then one figure on a single pdf.
   :param normalize: If True then the eigenvalues are normalized such that max |e_i| = 1
   """
   print(f'Loading {eig_file_name} (plot eig)')
   eig = load(eig_file_name)
   if normalize:
      eig /= numpy.max(numpy.abs(eig))

   if isinstance(plot_file, str):
      pdf = PdfPages(plot_file)
   elif isinstance(plot_file, PdfPages):
      pdf = plot_file
   else:
      raise Exception('Wrong plot file format')

   print(f'Plotting eigenvalues')
   plt.figure(figsize=(20, 10))
   plt.plot(real(eig), imag(eig), '.')
   plt.hlines(0, min(real(eig)), numpy.max(real(eig)), 'k')
   plt.vlines(0, min(imag(eig)), numpy.max(imag(eig)), 'k')
   pdf.savefig(bbox_inches='tight')
   plt.close()

def plot_eig_from_operator(A, plot_file, normalize=True):
   """
   Plot the eigenvalues of a matrix
   :param A: Linear operator
   :param plot_file: Path to the file where the plot will be saved. Can also be a PdfPages to have more than one figure
                     on a single pdf.
   :param normalize: If True then the eigenvalues are normalized such that max |e_i| = 1
   """

   print(f'Computing eigenvalues')
   num_vals = min(4000, A.shape[0] - 2)
   eig, _ = scipy.sparse.linalg.eigs(A, k=num_vals)

   if normalize:
      eig /= max(abs(eig))

   if isinstance(plot_file, str):
      pdf = PdfPages(plot_file)
   elif isinstance(plot_file, PdfPages): 
      pdf = plot_file
   else:
      raise Exception('Wrong plot file format')

   print(f'Plotting eigenvalues')
   plt.figure(figsize=(20, 10))
   plt.plot(real(eig), imag(eig), '.')
   plt.hlines(0, min(real(eig)), max(real(eig)), 'k')
   plt.vlines(0, min(imag(eig)), max(imag(eig)), 'k')
   pdf.savefig(bbox_inches='tight')
   plt.close()

def plot_spy(jac_file_name, plot_file, prec = 0):
   """
   Plot the spy of a matrix
   :param jac_file_name: Path to the file where the matrix is stored
   :param plot_file: Path to the file where the plot will be saved. Can also be a PdfPages to have more then one figure on a single pdf.
   :param prec: If precision is 0, any non-zero value will be plotted. Otherwise, values of |Z|>precision will be plotted.
   """
   print(f'Loading {jac_file_name} (plot spy)')
   J = load_npz(jac_file_name)

   if isinstance(plot_file, str):
      pdf = PdfPages(plot_file)
   elif isinstance(plot_file, PdfPages):
      pdf = plot_file
   else:
      raise Exception('Wrong plot file format')

   print(f'Plotting spy')
   plt.figure(figsize=(20, 20))
   plt.spy(J.toarray(), precision=prec)
   pdf.savefig(bbox_inches='tight')
   plt.close()

def output_dir(name):
   return os.path.join('./jacobian', name)

def jac_file(name, rhs):
   return os.path.join(output_dir(name), f'J_{rhs}.npz')

def eig_file(name, rhs):
   return os.path.join(output_dir(name), f'eig_{rhs}.npy')

def main(args):
   # rhs_type = ['all', 'exp', 'imp']
   rhs_type = ['all']
   name = args.name
   if args.gen_case is not None:
      config = args.gen_case
      if MPI.COMM_WORLD.rank == 0: os.makedirs(output_dir(name), exist_ok=True)

      for rhs in rhs_type:
         (Q, matvec, rhs_fun) = get_matvec(config, rhs, state_file=args.from_state_file)
         gen_matrix(Q, matvec, jac_file(name, rhs), permute=args.permute)

   print(f'Generation done')
   if args.plot and MPI.COMM_WORLD.rank == 0:
      pdf_spy = PdfPages('./jacobian/spy_' + name + '.pdf')
      pdf_eig = PdfPages('./jacobian/eig_' + name + '.pdf')

      for rhs in rhs_type:
         try:
            compute_eig(jac_file(name, rhs), eig_file(name, rhs), args.max_num_eigval)
            plot_eig(eig_file(name, rhs), pdf_eig)
            plot_spy(jac_file(name, rhs), pdf_spy)
         except(FileNotFoundError, IOError):
            print(f'Could not open file for case {name}, rhs {rhs}')
            pdf_spy.close()
            pdf_eig.close()
            raise

      pdf_spy.close()
      pdf_eig.close()

if __name__ == '__main__':

   parser = argparse.ArgumentParser(description='''Generate or plot system matrix and eigenvalues''')
   parser.add_argument('--gen-case', type=str, default=None, help='Generate a system matrix from given case file')
   parser.add_argument('--plot', action='store_true', help='Plot the given system matrix and its eigenvalues')
   parser.add_argument('--from-state-file', type = str, default = None,
                       help = 'Generate system matrix from a given state vector. (Still have to specify a config file)')
   parser.add_argument('--max-num-eigval', type = int, default = 0,
                       help = 'Maximum number of eigenvalues to compute (0 means all of them)')
   parser.add_argument('--permute', action='store_true', help='Permute the jacobian matrix so that all nodes/variables'
                       ' associated with an element form a block')
   parser.add_argument('name', type=str, help='Name of the case/system matrix')

   main(parser.parse_args())
