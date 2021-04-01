#!/usr/bin/env python3

import sys
import os
from itertools import product
from time import time
import argparse

import mpi4py
import numpy
from numpy import zeros, zeros_like, save, load, real, imag, hstack, max, abs #, vstack
from numpy.linalg import eigvals
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.sparse import csc_matrix, save_npz, load_npz, vstack
import scipy.sparse.linalg

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


def manual_spy(mat, ax):
   from scipy.sparse import coo_matrix
   from matplotlib.patches import Rectangle
   # if not isinstance(mat, coo_matrix):
   #    mat = coo_matrix(mat)
   for (x, y) in zip(mat.col, mat.row):
      ax.add_artist(Rectangle(
         xy=(x-0.5, y-0.5), width=1, height=1))
   ax.set_xlim(-0.5, mat.shape[1]-0.5)
   ax.set_ylim(-0.5, mat.shape[0]-0.5)
   ax.invert_yaxis()
   ax.set_aspect(float(mat.shape[0])/float(mat.shape[1]))


def get_matvec_sw(cfg_file, rhs):
   """
   Return the initial condition and function handle to compute the action of the Jacobian on a vector
   :param cfg_file: path to the configuration file
   :param rhs: type of rhs can be 'all', 'exp' or 'imp'
   :return: (Q, matvec)
   """
   param = Configuration(cfg_file)
   ptopo = Distributed_World()
   geom = cubed_sphere(param.nb_elements_horizontal, param.nb_elements_vertical, param.nbsolpts, param.λ0, param.ϕ0,
                       param.α0, param.ztop, ptopo)
   mtrx = DFR_operators(geom, param)
   metric = Metric(geom)
   Q, topo = initialize_sw(geom, metric, mtrx, param)

   if rhs == 'all':
      rhs = lambda q: rhs_sw(q, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal, param.case_number, param.filter_apply)
   elif rhs == 'exp':
      rhs = lambda q: rhs_sw_explicit(q, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal, param.case_number, param.filter_apply)
   elif rhs == 'imp':
      rhs = lambda q: rhs_sw_implicit(q, geom, mtrx, metric, topo, ptopo, param.nbsolpts, param.nb_elements_horizontal, param.case_number, param.filter_apply)
   else:
      raise Exception('Wrong rhs name')

   # return (Q, lambda v: matvec_fun(v, 1, Q, rhs), rhs)
   return Q, lambda v: matvec_rat(v, param.dt, Q, rhs), rhs


def gen_matrix(Q, matvec, rhs_fun, jac_file, rhs_file):
   """
   Compute and store the Jacobian matrix
   :param Q: Solution vector where the Jacobian is computed
   :param matvec: Function handle to compute the action of the jacobian on a vector
   :param jac_file: Path to the file where the jacobian will be stored
   """
   neq, ni, nj = Q.shape
   n_loc = Q.size

   rank = mpi4py.MPI.COMM_WORLD.Get_rank()
   size = mpi4py.MPI.COMM_WORLD.Get_size()

   Qid = zeros_like(Q)
   # J = zeros((n_loc, size*n_loc))
   J = csc_matrix((n_loc, size*n_loc), dtype=Q.dtype)

   idx = 0
   total_num_iter = size * n_loc

   t0 = 0.0
   if rank == 0:
      print('')
      t0 = time()

   def print_progress(current, total, t_init):
      remaining = (time() - t_init) * (float(total) / current - 1.0)
      print(f'\r {current * 100.0 / total : 5.1f}% ({remaining: 4.0f}s)', end=' ')

   for r in range(size):
      for (i, j, k) in product(range(neq), range(ni), range(nj)):
         if rank == r:
            Qid[i, j, k] = 1.0

         col = csc_matrix(matvec(Qid.flatten())).transpose()
         J[:, idx] = col
         idx += 1
         Qid[i, j, k] = 0.0

         if rank == 0:
            print_progress(idx, total_num_iter, t0)

   J_comm = mpi4py.MPI.COMM_WORLD.gather(J, root=0)
   rhs_comm = mpi4py.MPI.COMM_WORLD.gather(rhs_fun(Q).flatten() / 1800.0, root=0)
   Q_comm = mpi4py.MPI.COMM_WORLD.gather(Q.flatten(), root=0)

   if rank == 0:
      print('')
      glb_J = vstack(J_comm)
      # save(jac_file, glb_J)
      save_npz(jac_file, glb_J)

      glb_rhs = hstack(rhs_comm)
      save(rhs_file, glb_rhs)

      glb_Q = hstack(Q_comm)

      # print(f'Q: \n{Q[0]}')
      # print(f'rhs: \n{rhs_comm[0]}')
      # print(f'shape Q.flat: {glb_Q.shape}, shape J: {glb_J.shape}')
      # print(f'Ax: \n{glb_J @ glb_Q}')


def compute_eig(jac_file, eig_file):
   """
   Compute and save the eigenvalues of a matrix
   :param jac_file: Path to the file where the matrix is stored
   :param eig_file: Path to the file where the eigenvalues will be stored
   """
   print(f'loading {jac_file}')
   J = load_npz(f'{jac_file}.npz')
   print(f'Computing eigenvalues')
   if J.shape[0] < 10000:
      eig = eigvals(J.toarray())
   else:
      num_vals = min(4000, J.shape[0] - 2)
      eig, _ = scipy.sparse.linalg.eigs(J, k=num_vals)
   print(f'Saving {eig_file}')
   save(eig_file, eig)


def plot_eig(eig_file, plot_file, normalize=True):
   """
   Plot the eigenvalues of a matrix
   :param eig_file: Path to the file where the eigenvalues are stored
   :param plot_file: Path to the file where the plot will be saved. Can also be a PdfPages to have more then one figure on a single pdf.
   :param normalize: If True then the eigenvalues are normalized such that max |e_i| = 1
   """
   print(f'loading {eig_file}')
   eig = load(eig_file)
   if normalize:
      eig /= max(abs(eig))

   if type(plot_file) == str:
      pdf = PdfPages(plot_file)
   elif type(plot_file) == PdfPages: 
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


def plot_spy(jac_file, plot_file, prec = 0):
   """
   Plot the spy of a matrix
   :param jac_file: Path to the file where the matrix is stored
   :param plot_file: Path to the file where the plot will be saved. Can also be a PdfPages to have more then one figure on a single pdf.
   :param prec: If precision is 0, any non-zero value will be plotted. Otherwise, values of |Z|>precision will be plotted.
   """
   print(f'Loading {jac_file}')
   J = load_npz(f'{jac_file}.npz')

   if type(plot_file) == str:
      pdf = PdfPages(plot_file)
   elif type(plot_file) == PdfPages:
      pdf = plot_file
   else:
      raise Exception('Wrong plot file format')

   print(f'Plotting spy')
   plt.figure(figsize=(20, 20))
   if J.shape[0] < 10000:
      plt.spy(J.toarray(), precision=prec)
   else:
      manual_spy(J, plt.gca())
   pdf.savefig(bbox_inches='tight')
   plt.close()


def main(args):
   rhs_type = ['all', 'exp', 'imp']
   if args.gen_case is not None:
      config = args.gen_case
      name = args.name
      os.makedirs(f'./jacobian/{name}/', exist_ok=True)

      for rhs in rhs_type:
         (Q, matvec, rhs_fun) = get_matvec_sw(config, rhs)
         gen_matrix(Q, matvec, rhs_fun, f'./jacobian/{name}/J_{rhs}', f'./jacobian/{name}/initial_rhs_{rhs}.npy')
   elif args.plot:
      name = args.name
      pdf_spy = PdfPages('./jacobian/spy_' + name + '.pdf')
      pdf_eig = PdfPages('./jacobian/eig_' + name + '.pdf')

      for rhs in rhs_type:
         try:
            jac_file = f'./jacobian/{name}/J_{rhs}'
            eig_file = f'./jacobian/{name}/eig_{rhs}.npy'
            compute_eig(jac_file, eig_file)
            plot_eig(eig_file, pdf_eig)
            plot_spy(jac_file, pdf_spy)
         except(FileNotFoundError, IOError):
            print(f'Could not open file for case {name}, rhs {rhs}')

      pdf_spy.close()
      pdf_eig.close()


if __name__ == '__main__':

   parser = argparse.ArgumentParser(description='''Generate or plot system matrix and eigenvalues\njoijoi''')
   command = parser.add_mutually_exclusive_group()
   command.add_argument('--gen-case', type=str, default=None, help='Generate a system matrix from given case file (need to run it with 6 processes')
   command.add_argument('--plot', action='store_true', help='Plot the given system matrix and its eigenvalues')
   parser.add_argument('name', type=str, help='Name of the case/system matrix')

   main(parser.parse_args())
