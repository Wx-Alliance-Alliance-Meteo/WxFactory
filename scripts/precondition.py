#!/usr/bin/env python

import argparse
import numpy as np
import os
from time import time
from functools import partial

import scipy
import scipy.sparse.linalg
from scipy.sparse import load_npz

import pyamg
import linsol

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
   case_dir = f'{script_dir}/jacobian/{case_name}'
   os.makedirs(case_dir, exist_ok=True)

   jacobian_file = f'{case_dir}/J_all.npz'
   rhs_file = f'{case_dir}/initial_rhs_all.npy'

   try:
      jacobian = load_npz(jacobian_file)
      initial_rhs = np.load(rhs_file)
   except FileNotFoundError:
      print(f'Could not open file(s) for "{case_name}".')
      raise

   # print(f'jacobian = {jacobian}')
   # print(f'initial rhs = {initial_rhs}')

   return scipy.sparse.csc_matrix(jacobian), initial_rhs


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

   result, solve_time = timed_call(partial(linsol.fgmres, matvec, rhs, restart=20, maxiter=25, tol=tolerance,
                                           preconditioner=precond))

   print_result(jacobian, rhs, result[0], result[2], solve_time, prefix='GEF FGMRES')
   return result[0]

def solve_pyamg_gmres(solver, jacobian, rhs, precond=None):

   reset_iter()
   def gmres_callback(x):
      increment_iter()

   result, solve_time = timed_call(
      partial(solver, jacobian, rhs, M=precond, restrt=20, maxiter=25, tol=tolerance, callback=gmres_callback))

   print_result(jacobian, rhs, result[0], g_num_iter, solve_time, prefix=f'pyAMG {solver.__name__}')
   return result[0]

def solve_pyamg_krylov(solver, jacobian, rhs, x0=None, precond=None):
   reset_iter()
   def pyamg_callback(x):
      increment_iter()

   result, t = timed_call(partial(solver, jacobian, rhs, tol=tolerance, x0=x0, maxiter=500, M=precond, callback=pyamg_callback))
   print_result(jacobian, rhs, result[0], g_num_iter, t, prefix=f'pyAMG {solver.__name__}')
   return result[0]

def solve_gmres(jacobian, rhs, precond=None):
   reset_iter()
   def gmres_callback(x):
      increment_iter()

   result, solve_time = timed_call(
      partial(scipy.sparse.linalg.gmres, jacobian, rhs, M=precond, restart=20, tol=tolerance, maxiter=500,
              callback=gmres_callback))

   print_result(jacobian, rhs, result[0], g_num_iter, solve_time, prefix='Scipy GMRES')
   return result[0]

def solve_lgmres(jacobian, rhs, precond=None):

   reset_iter()
   def lgmres_callback(x):
      increment_iter()

   num_inner_it = 20

   result, solve_time = timed_call(
      partial(scipy.sparse.linalg.lgmres, jacobian, rhs, M=precond, tol=tolerance, atol=tolerance * 0.1,
              inner_m=num_inner_it, maxiter=200, callback=lgmres_callback))

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


def main(args):
   jacobian, initial_rhs = load_matrix(args.case_name)

   sol_gef    = solve_gef_fgmres(jacobian, initial_rhs)
   sol_scipy  = solve_gmres(jacobian, initial_rhs)
   # sol_lgmres = solve_lgmres(jacobian, initial_rhs)
   sol_amg    = solve_pyamg_gmres(pyamg.krylov.gmres, jacobian, initial_rhs)
   solve_pyamg_gmres(pyamg.krylov.fgmres, jacobian, initial_rhs)
   # solve_pyamg_krylov(pyamg.krylov.cg, jacobian, initial_rhs)
   # solve_pyamg_krylov(pyamg.krylov.bicgstab, jacobian, initial_rhs)
   # solve_pyamg_krylov(pyamg.krylov.cgne, jacobian, initial_rhs)

   # solve_jacobi(jacobian, initial_rhs)

   # solve_gmres_diag_precond(jacobian, initial_rhs)
   # solve_gmres_diag_precond(jacobian, initial_rhs, 2)
   # solve_gmres_diag_precond(jacobian, initial_rhs, 3)
   # solve_gmres_diag_precond(jacobian, initial_rhs, 4)
   # solve_gmres_ilu_precond(jacobian, initial_rhs)

   # solve_lgmres_diag_precond(jacobian, initial_rhs)
   # solve_lgmres_diag_precond(jacobian, initial_rhs, 2)
   # solve_lgmres_diag_precond(jacobian, initial_rhs, 3)
   # solve_lgmres_diag_precond(jacobian, initial_rhs, 4)
   # solve_lgmres_ilu_precond(jacobian, initial_rhs)

   A = scipy.sparse.csr_matrix(jacobian)
   # A = pyamg.gallery.poisson((jacobian.shape[0],), format='csr')
   # print(f'jacobian shape: {jacobian.shape}, A shape: {A.shape}')

   print('Rootnode preconditioner... ', end='')
   mg_solver_rootnode = pyamg.rootnode_solver(A, max_levels=2)
   precond_rootnode   = mg_solver_rootnode.aspreconditioner(cycle='V')
   print(' initialized.')

   solve_gef_fgmres(A, initial_rhs, precond=precond_rootnode)
   solve_gmres(A, initial_rhs, precond=precond_rootnode)
   solve_pyamg_gmres(pyamg.krylov.gmres, A, initial_rhs, precond=precond_rootnode)
   solve_pyamg_gmres(pyamg.krylov.fgmres, A, initial_rhs, precond=precond_rootnode)
   # solve_pyamg_krylov(pyamg.krylov.cg, A, initial_rhs, precond=precond_rootnode)
   # solve_pyamg_krylov(pyamg.krylov.bicgstab, A, initial_rhs, precond=precond_rootnode)
   # solve_pyamg_krylov(pyamg.krylov.cgne, A, initial_rhs, precond=precond_rootnode)

   print('Smoothed aggregation preconditioner... ', end='')
   mg_solver_sas = pyamg.smoothed_aggregation_solver(A, max_levels=2)
   precond_sas   = mg_solver_sas.aspreconditioner(cycle='V')
   print(' initialized.')
   
   solve_gef_fgmres(A, initial_rhs, precond=precond_sas)
   solve_gmres(A, initial_rhs, precond=precond_sas)
   solve_pyamg_gmres(pyamg.krylov.gmres, A, initial_rhs, precond=precond_sas)
   solve_pyamg_gmres(pyamg.krylov.fgmres, A, initial_rhs, precond=precond_sas)
   # solve_pyamg_krylov(pyamg.krylov.cg, A, initial_rhs, precond=precond_sas)
   # solve_pyamg_krylov(pyamg.krylov.bicgstab, A, initial_rhs, precond=precond_sas)
   # solve_pyamg_krylov(pyamg.krylov.cgne, A, initial_rhs, precond=precond_sas)

   gef_sci_diff = sol_scipy - sol_gef
   norm = scipy.linalg.norm(gef_sci_diff)
   print(f'Sci-GEF diff:    {norm:.3e}, rel {norm/scipy.linalg.norm(sol_scipy):.3e}')
   print(f'Sci-AMG diff:    {scipy.linalg.norm(sol_scipy - sol_amg):.3e}')
   # print(f'Sci-Lgmres diff: {scipy.linalg.norm(sol_scipy - sol_lgmres):.3e}')


if __name__ == '__main__':
   np.set_printoptions(precision=2)

   parser = argparse.ArgumentParser(description='''Compute a preconditioner for a given input matrix''')
   parser.add_argument('case_name', help='Name of the .npy file where the matrix is stored')
   parser.add_argument('--config', default=f'{script_dir}/test_case.ini')

   main(parser.parse_args())
