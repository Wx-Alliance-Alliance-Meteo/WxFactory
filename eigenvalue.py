import sys
import os
from itertools import product
from time import time

import mpi4py
import numpy
from numpy import zeros, zeros_like, save, load, real, imag, hstack, max, abs #, vstack
from numpy.linalg import eigvals
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.sparse import csc_matrix, save_npz, load_npz, vstack

from program_options import Configuration
from cubed_sphere import cubed_sphere
from initialize import initialize_sw
from matrices import DFR_operators
from matvec import matvec_fun, matvec_rat
from metric import Metric
from parallel import Distributed_World
from rhs_sw import rhs_sw
from rhs_sw_explicit import rhs_sw_explicit
from rhs_sw_implicit import rhs_sw_implicit

def get_matvec_sw(cfg_file, rhs):
   """
   Return the initial condition and function handle to compute the action of the Jacobian on a vector
   :param cfg_file: path to the configuration file
   :param rhs: type of rhs can be 'all', 'exp' or 'imp'
   :return: (Q, matvec)
   """
   param = Configuration(cfg_file)
   ptopo = Distributed_World()
   geom = cubed_sphere(param.nb_elements, param.nbsolpts, param.λ0, param.ϕ0, param.α0, ptopo)
   mtrx = DFR_operators(geom, param)
   metric = Metric(geom)
   Q, topo = initialize_sw(geom, metric, mtrx, param)

   if rhs == 'all':
      rhs = lambda q: rhs_sw(q, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements, param.case_number, param.filter_apply)
   elif rhs == 'exp':
      rhs = lambda q: rhs_sw_explicit(q, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements, param.case_number, param.filter_apply)
   elif rhs == 'imp':
      rhs = lambda q: rhs_sw_implicit(q, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements, param.case_number, param.filter_apply)
   else:
      raise Exception('Wrong rhs name')

   # return (Q, lambda v: matvec_fun(v, 1, Q, rhs), rhs)
   return (Q, lambda v: matvec_rat(v, 1800.0, Q, rhs), rhs)


def gen_matrix(Q, matvec, rhs_fun, jac_file, rhs_file):
   """
   Compute and store the Jacobian matrix
   :param Q: Solution vector where the Jacobian is computed
   :param matvec: Function handle to compute the action of the jacobian on a vector
   :param jac_file: Path to the file where the jacobian will be stored
   """
   neq,ni,nj = Q.shape
   n_loc = Q.size

   rank = mpi4py.MPI.COMM_WORLD.Get_rank()
   size = mpi4py.MPI.COMM_WORLD.Get_size()

   Qid = zeros_like(Q)
   # J = zeros((n_loc, size*n_loc))
   J = csc_matrix((n_loc, size*n_loc), dtype = Q.dtype)

   idx = 0
   total_num_iter = size * n_loc

   if rank == 0:
      print('')
      t0 = time()

   def print_progress(current, total, t0):
      remaining = (time() - t0) * (float(total) / current - 1.0)
      print(f'\r {current * 100.0 / total : 5.1f}% ({remaining: 4.0f}s)', end = ' ')

   for r in range(size):
      for (i,j,k) in product(range(neq),range(ni),range(nj)):
         if rank == r:
            Qid[i,j,k] = 1.0

         col = csc_matrix(matvec(Qid.flatten())).transpose()
         J[:, idx] = col
         idx += 1
         Qid[i, j, k] = 0.0

         if rank == 0:
            print_progress(idx, total_num_iter, t0)

   J_comm = mpi4py.MPI.COMM_WORLD.gather(J, root=0)
   rhs_comm = mpi4py.MPI.COMM_WORLD.gather(rhs_fun(Q).flatten(), root=0)

   if rank == 0:
      print('')
      glb_J = vstack(J_comm)
      # save(jac_file, glb_J)
      save_npz(jac_file, glb_J)

      glb_rhs = hstack(rhs_comm)
      save(rhs_file, glb_rhs)


def compute_eig(jac_file, eig_file):
   """
   Compute and save the eigenvalues of a matrix
   :param jac_file: Path to the file where the matrix is stored
   :param eig_file: Path to the file where the eigenvalues will be stored
   """
   J = load_npz(f'{jac_file}.npz').toarray()
   eig = eigvals(J)
   save(eig_file, eig)


def plot_eig(eig_file, plot_file, normalize = True):
   """
   Plot the eigenvalues of a matrix
   :param eig_file: Path to the file where the eigenvalues are stored
   :param plot_file: Path to the file where the plot will be saved. Can also be a PdfPages to have more then one figure on a single pdf.
   :param normalize: If True then the eigenvalues are normalized such that max |e_i| = 1
   """
   eig = load(eig_file)
   if normalize: eig /= max(abs(eig))

   if type(plot_file) == str:
      pdf = PdfPages(plot_file)
   elif type(plot_file) == PdfPages: 
      pdf =plot_file
   else:
      raise Exception('Wrong plot file format')

   plt.figure(figsize=(20, 10))
   plt.plot(real(eig), imag(eig), '.')
   plt.hlines(0, min(real(eig)), max(real(eig)), 'k')
   plt.vlines(0, min(imag(eig)), max(imag(eig)), 'k')
   pdf.savefig(bbox_inches='tight')
   plt.close()


def plot_spy(jac_file, plot_file, prec = 0):
   """
   Plot the spy of a matrix
   :param jac_file: Path to the file where the matrix is stored
   :param plot_file: Path to the file where the plot will be saved. Can also be a PdfPages to have more then one figure on a single pdf.
   :param prec: If precision is 0, any non-zero value will be plotted. Otherwise, values of |Z|>precision will be plotted.
   """
   J = load_npz(f'{jac_file}.npz').toarray()

   if type(plot_file) == str:
      pdf = PdfPages(plot_file)
   elif type(plot_file) == PdfPages:
      pdf =plot_file
   else:
      raise Exception('Wrong plot file format')

   plt.figure(figsize=(20, 20))
   plt.spy(J, precision=prec)
   pdf.savefig(bbox_inches='tight')
   plt.close()


def main():
   rhs_type = ['all', 'exp', 'imp']
   if sys.argv[1] == 'gen':
      config = sys.argv[2]
      name = sys.argv[3]
      os.makedirs(f'./jacobian/{name}/', exist_ok=True)
      for rhs in rhs_type:
         (Q, matvec, rhs_fun) = get_matvec_sw(config, rhs)
         gen_matrix(Q, matvec, rhs_fun, f'./jacobian/{name}/J_{rhs}', f'./jacobian/{name}/initial_rhs_{rhs}.npy')
   elif sys.argv[1] == 'plot':
      name = sys.argv[2]
      pdf_spy = PdfPages('./jacobian/spy_' + name + '.pdf')
      pdf_eig = PdfPages('./jacobian/eig_' + name + '.pdf')

      for rhs in rhs_type:
         jac_file = f'./jacobian/{name}/J_{rhs}'
         eig_file = f'./jacobian/{name}/eig_{rhs}.npy'
         compute_eig(jac_file, eig_file)
         plot_eig(eig_file, pdf_eig)
         plot_spy(jac_file, pdf_spy)

      pdf_spy.close()
      pdf_eig.close()


if __name__ == '__main__':
   if len(sys.argv) < 2:
      print("Usage: ")
      print("   - Generate the jacobian matrices:")
      print("mpirun -n 6 python eigenvalue.py gen config.ini case_name")
      print("   - Plot the eigenvalues:")
      print("python eigenvalue.py plot case_name")
   else:
      main()
