#!/usr/bin/env python

import argparse
import numpy as np
import os
import sys
from time import time
from functools import partial

import scipy
import scipy.sparse.linalg
from scipy.sparse import load_npz

try:
   import pyamg
   has_pyamg = True
except ModuleNotFoundError:
   has_pyamg = False
   print(f'PyAMG not installed. Won\'t do the associated tests')

# We assume the script is in a subfolder of the main project
main_gef_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(main_gef_dir)

from solvers import fgmres

script_dir = os.path.dirname(os.path.abspath(__file__))

g_num_iter = 0
def increment_iter():
   global g_num_iter
   g_num_iter = g_num_iter + 1

def reset_iter():
   global g_num_iter
   g_num_iter = 0


tolerance = 1e-7

def load_matrix(case_name):
   print(f'Reading matrix for "{case_name}"')
   case_dir = f'{main_gef_dir}/jacobian/{case_name}'
   os.makedirs(case_dir, exist_ok=True)

   files = []
   matrices = []
   for suffix in ['', '.p']:
      jacobian_file = f'{case_dir}/J_all{suffix}.npz'
      rhs_file = f'{case_dir}/rhs_all{suffix}.npy'
      files.append((jacobian_file, rhs_file))

   for j_file, r_file in files:
      try:
         print(f'Loading {j_file}')
         jacobian = load_npz(j_file)
         initial_rhs = np.load(r_file)
         matrices.append((scipy.sparse.csc_matrix(jacobian), initial_rhs))
      except FileNotFoundError:
         print(f'Could not open file(s) for "{case_name}".')

   return matrices


def timed_call(function):
   t0 = time()
   result = function()
   t1 = time()

   return result, t1 - t0


def print_result(jacobian, rhs, sol, num_iter, solve_time, prefix=''):
   Ax = jacobian @ sol
   abs_error = scipy.linalg.norm(rhs - Ax)
   rel_error = abs_error / scipy.linalg.norm(rhs)

   comment = ''
   if rel_error > tolerance * 1000:
      comment = ' -- NOT WORKING'
   elif rel_error > tolerance * 100:
      comment = ' -- that\'s pretty bad'
   elif rel_error > tolerance:
      comment = ' -- not so good'

   print(f'{prefix:15s} -- Relative error: {rel_error:.3e}, absolute: {abs_error:.3e}, '
         f'time {solve_time:.3f} s, num iterations: {num_iter}'
         f'{comment}')

def solve_gef_fgmres(jacobian, rhs, precond=None):

   def matvec(vec):
      return jacobian @ vec

   result, solve_time = timed_call(partial(fgmres, matvec, rhs, restart=200, maxiter=25, tol=tolerance,
                                           preconditioner=precond))

   print_result(jacobian, rhs, result[0], result[3], solve_time, prefix='GEF FGMRES')
   return result[0]

def solve_pyamg_gmres(solver, jacobian, rhs, precond=None):

   reset_iter()
   def gmres_callback(x):
      increment_iter()

   result, solve_time = timed_call(
      partial(solver, jacobian, rhs, M=precond, restrt=200, maxiter=25, tol=tolerance, callback=gmres_callback))

   print_result(jacobian, rhs, result[0], g_num_iter, solve_time, prefix=f'pyAMG {solver.__name__}')
   return result[0]

def solve_pyamg_krylov(solver, jacobian, rhs, x0=None, precond=None):
   reset_iter()
   def pyamg_callback(x):
      increment_iter()

   result, t = timed_call(partial(solver, jacobian, rhs, tol=tolerance, x0=x0, maxiter=5000, M=precond, callback=pyamg_callback))
   print_result(jacobian, rhs, result[0], g_num_iter, t, prefix=f'pyAMG {solver.__name__}')
   return result[0]

def solve_gmres(jacobian, rhs, precond=None):
   reset_iter()
   def gmres_callback(x):
      increment_iter()

   result, solve_time = timed_call(
      partial(scipy.sparse.linalg.gmres, jacobian, rhs, M=precond, restart=200, tol=tolerance, maxiter=5000,
              callback=gmres_callback))

   print_result(jacobian, rhs, result[0], g_num_iter, solve_time, prefix='Scipy GMRES')
   return result[0]

def solve_lgmres(jacobian, rhs, precond=None):

   reset_iter()
   def lgmres_callback(x):
      increment_iter()

   num_inner_it = 200

   result, solve_time = timed_call(
      partial(scipy.sparse.linalg.lgmres, jacobian, rhs, M=precond, tol=tolerance, atol=tolerance * 0.1,
              inner_m=num_inner_it, maxiter=25, callback=lgmres_callback))

   print_result(jacobian, rhs, result[0], f'{g_num_iter}*{num_inner_it}', solve_time, prefix='Scipy LGMRES')
   return result[0]


def get_diag(matrix, num_stages = 1):
   diagonal_mat = scipy.sparse.diags([matrix[i, i] for i in range(matrix.shape[0])], format='csc')

   for d in range(1, num_stages):
      diagonal_mat =\
         diagonal_mat + \
         scipy.sparse.diags([matrix[i+d, i] for i in range(matrix.shape[0] - d)], offsets=-d, format='csc') + \
         scipy.sparse.diags([matrix[i, i+d] for i in range(matrix.shape[0] - d)], offsets=d, format='csc')

   return diagonal_mat


def solve_gmres_diag_precond(jacobian, rhs, num_stages=1):
   print(f'Using a diagonal preconditioner with {(num_stages - 1) * 2 + 1} diagonals')

   diagonal_mat = get_diag(jacobian, num_stages)
   diag_inv = scipy.sparse.linalg.inv(diagonal_mat)

   # print(f'diag mat: {diagonal_mat.toarray()}')
   solve_gmres(jacobian, rhs, precond=diag_inv)


def solve_lgmres_diag_precond(jacobian, rhs, num_stages=1):
   print(f'Using a diagonal preconditioner with {(num_stages - 1) * 2 + 1} diagonals')

   diagonal_mat = get_diag(jacobian, num_stages)
   diag_inv = scipy.sparse.linalg.inv(diagonal_mat)

   # print(f'diag mat: {diagonal_mat.toarray()}')
   solve_lgmres(jacobian, rhs, precond=diag_inv)


def get_ilu_inverse(jacobian):
   inv_approx_lu = scipy.sparse.linalg.spilu(jacobian, drop_tol=1e-5, fill_factor=15)
   # inv_approx_lu = scipy.sparse.linalg.splu(jacobian)

   n_row = jacobian.shape[0]
   # Pr = scipy.sparse.csc_matrix((np.ones(n_row), (inv_approx_lu.perm_r, np.arange(n_row))))
   # Pc = scipy.sparse.csc_matrix((np.ones(n_row), (np.arange(n_row), inv_approx_lu.perm_c)))
   # inv_L = scipy.sparse.linalg.inv(inv_approx_lu.L)
   # inv_U = scipy.sparse.linalg.inv(inv_approx_lu.U)
   # inv_approx = Pc @ inv_U @ inv_L @ Pr
   inv_approx = inv_approx_lu.solve(scipy.sparse.identity(n_row).toarray())

   return inv_approx


def solve_gmres_ilu_precond(jacobian, rhs):
   print('Using an ILU preconditioner')
   inv_approx = get_ilu_inverse(jacobian)
   solve_gmres(jacobian, rhs, precond=inv_approx)


def solve_lgmres_ilu_precond(jacobian, rhs):
   print('Using an ILU preconditioner')
   inv_approx = get_ilu_inverse(jacobian)
   solve_lgmres(jacobian, rhs, precond=inv_approx)


def solve_jacobi(jacobian, rhs):
   L = scipy.sparse.tril(jacobian, -1)
   U = scipy.sparse.triu(jacobian, 1)
   D = jacobian - L - U
   for i in range(D.shape[0]):
      D[i, i] = 1.0 / D[i, i]

   tol_rel = scipy.linalg.norm(rhs) * tolerance

   x = np.zeros_like(rhs)
   max_it = 500
   time_0 = time()
   for i in range(max_it):
      x = D @ (rhs - (L + U) @ x)
      r = rhs - jacobian @ x
      if scipy.linalg.norm(r) < tol_rel: break
   time_1 = time()

   t = time_1 - time_0

   print_result(jacobian, rhs, x, i, t, prefix='Jacobi')
   return x, t

def run_set(jac, rhs):

   sol_gef    = solve_gef_fgmres(jac, rhs)
   sol_scipy  = solve_gmres(jac, rhs)
   # sol_lgmres = solve_lgmres(jac, rhs)
   if has_pyamg:
      sol_amg    = solve_pyamg_gmres(pyamg.krylov.gmres, jac, rhs)
      solve_pyamg_gmres(pyamg.krylov.fgmres, jac, rhs)
      # solve_pyamg_krylov(pyamg.krylov.cg, jac, rhs)
      # solve_pyamg_krylov(pyamg.krylov.bicgstab, jac, rhs)
      # solve_pyamg_krylov(pyamg.krylov.cgne, jac, rhs)

   # solve_jacobi(jac, rhs)

   # solve_gmres_diag_precond(jac, rhs)
   # solve_gmres_diag_precond(jac, rhs, 2)
   # solve_gmres_diag_precond(jac, rhs, 3)
   # solve_gmres_diag_precond(jac, rhs, 4)
   # solve_gmres_ilu_precond(jac, rhs)

   # solve_lgmres_diag_precond(jac, rhs)
   # solve_lgmres_diag_precond(jac, rhs, 2)
   # solve_lgmres_diag_precond(jac, rhs, 3)
   # solve_lgmres_diag_precond(jac, rhs, 4)
   # solve_lgmres_ilu_precond(jac, rhs)

   A = scipy.sparse.csr_matrix(jac)
   # A = pyamg.gallery.poisson((jac.shape[0],), format='csr')
   # print(f'jac shape: {jac.shape}, A shape: {A.shape}')

   if has_pyamg:
      print('Rootnode preconditioner... ', end='')
      mg_solver_rootnode = pyamg.rootnode_solver(A, max_levels=2)
      precond_rootnode   = mg_solver_rootnode.aspreconditioner(cycle='V')
      print(' initialized.')

      solve_gef_fgmres(A, rhs, precond=precond_rootnode)
      solve_gmres(A, rhs, precond=precond_rootnode)
      solve_pyamg_gmres(pyamg.krylov.gmres, A, rhs, precond=precond_rootnode)
      solve_pyamg_gmres(pyamg.krylov.fgmres, A, rhs, precond=precond_rootnode)
      # solve_pyamg_krylov(pyamg.krylov.cg, A, rhs, precond=precond_rootnode)
      # solve_pyamg_krylov(pyamg.krylov.bicgstab, A, rhs, precond=precond_rootnode)
      # solve_pyamg_krylov(pyamg.krylov.cgne, A, rhs, precond=precond_rootnode)

      print('Smoothed aggregation preconditioner... ', end='')
      mg_solver_sas = pyamg.smoothed_aggregation_solver(A, max_levels=2)
      precond_sas   = mg_solver_sas.aspreconditioner(cycle='V')
      print(' initialized.')
   
      solve_gef_fgmres(A, rhs, precond=precond_sas)
      solve_gmres(A, rhs, precond=precond_sas)
      solve_pyamg_gmres(pyamg.krylov.gmres, A, rhs, precond=precond_sas)
      solve_pyamg_gmres(pyamg.krylov.fgmres, A, rhs, precond=precond_sas)
      # solve_pyamg_krylov(pyamg.krylov.cg, A, rhs, precond=precond_sas)
      # solve_pyamg_krylov(pyamg.krylov.bicgstab, A, rhs, precond=precond_sas)
      # solve_pyamg_krylov(pyamg.krylov.cgne, A, rhs, precond=precond_sas)

   gef_sci_diff = sol_scipy - sol_gef
   norm = scipy.linalg.norm(gef_sci_diff)
   print(f'Sci-GEF diff:    {norm:.3e}, rel {norm/scipy.linalg.norm(sol_scipy):.3e}')
   if has_pyamg:
      print(f'Sci-AMG diff:    {scipy.linalg.norm(sol_scipy - sol_amg):.3e}')
   # print(f'Sci-Lgmres diff: {scipy.linalg.norm(sol_scipy - sol_lgmres):.3e}')

def main(args):
   matrices = load_matrix(args.case_name)

   for jac, rhs in matrices:
      run_set(jac, rhs)


if __name__ == '__main__':
   np.set_printoptions(precision=2)

   parser = argparse.ArgumentParser(description='''Compute a preconditioner for a given input matrix''')
   parser.add_argument('case_name', help='Name of the .npy file where the matrix is stored')
   parser.add_argument('--config', default=f'{script_dir}/test_case.ini')

   main(parser.parse_args())
