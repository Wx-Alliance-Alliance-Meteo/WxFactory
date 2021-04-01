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

num_iter = 0
def increment_iter():
   global num_iter
   num_iter = num_iter + 1

def reset_iter():
   global num_iter
   num_iter = 0


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


def run_solver(function, jacobian, rhs):
   t0 = time()
   result = function()
   t1 = time()

   sol = result[0]

   Ax = jacobian @ sol
   abs_error = scipy.linalg.norm(rhs - Ax)
   rel_error = abs_error / scipy.linalg.norm(rhs)
   print(f'Relative error: {rel_error:.3e}, absolute: {abs_error:.3e}, time {t1 - t0:.3f} s')

   return sol


def solve_gef_fgmres(jacobian, rhs, inv_precond=None):

   def matvec(vec):
      return jacobian @ vec

   return run_solver(partial(linsol.fgmres, matvec, rhs, restart=20, tol=tolerance), jacobian, rhs)

def solve_pyamg_gmres(solver, jacobian, rhs, inv_precond=None):

   reset_iter()
   def gmres_callback(x):
      increment_iter()

   return run_solver(partial(solver, jacobian, rhs,
                             M=inv_precond, restrt=20, maxiter=20, tol=tolerance, callback=gmres_callback),
                     jacobian, rhs)

def solve_gmres(jacobian, rhs, inv_precond=None):

   reset_iter()
   def gmres_callback(x):
      increment_iter()

   return run_solver(partial(scipy.sparse.linalg.gmres, jacobian, rhs,
                             M=inv_precond, restart=20, tol=tolerance, maxiter=2000, callback=gmres_callback),
                     jacobian, rhs)

def solve_lgmres(jacobian, rhs, inv_precond = None):

   reset_iter()
   def lgmres_callback(x):
      increment_iter()

   return run_solver(partial(scipy.sparse.linalg.lgmres, jacobian, rhs, M=inv_precond,
                             tol=tolerance, atol=tolerance * 0.1, inner_m=20, maxiter=200, callback=lgmres_callback),
                     jacobian, rhs)


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
   solve_gmres(jacobian, rhs, inv_precond=diag_inv)


def solve_lgmres_diag_precond(jacobian, rhs, num_stages=1):
   print(f'Using a diagonal preconditioner with {(num_stages - 1) * 2 + 1} diagonals')

   diagonal_mat = get_diag(jacobian, num_stages)
   diag_inv = scipy.sparse.linalg.inv(diagonal_mat)

   # print(f'diag mat: {diagonal_mat.toarray()}')
   solve_lgmres(jacobian, rhs, inv_precond=diag_inv)


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
   solve_gmres(jacobian, rhs, inv_precond=inv_approx)


def solve_lgmres_ilu_precond(jacobian, rhs):
   print('Using an ILU preconditioner')
   inv_approx = get_ilu_inverse(jacobian)
   solve_lgmres(jacobian, rhs, inv_precond=inv_approx)


def main(args):
   jacobian, initial_rhs = load_matrix(args.case_name)

   sol_scipy = solve_gmres(jacobian, initial_rhs)
   sol_gef   = solve_gef_fgmres(jacobian, initial_rhs)
   sol_amg   = solve_pyamg_gmres(pyamg.krylov.gmres, jacobian, initial_rhs)
   sol_amg   = solve_pyamg_gmres(pyamg.krylov.fgmres, jacobian, initial_rhs)
   # sol_amg   = run_solver(partial(pyamg.krylov.cg, jacobian, initial_rhs, tol=tolerance), jacobian, initial_rhs)

   # pyamg_solver = pyamg.aggregation.smoothed_aggregation_solver(jacobian)
   # pyamg_solver = pyamg.aggregation.energy_prolongation_smoother(jacobian)
   # pyamg_solver = pyamg.aggregation.rootnode_solver(jacobian)

   # A = scipy.sparse.csr_matrix(jacobian)
   # C = pyamg.strength.classical_strength_of_connection(A)
   # splitting = pyamg.classical.split.RS(A)
   # P = pyamg.classical.direct_interpolation(A, C, splitting)
   # R = P.T
   #
   # levels = []
   # levels.append(pyamg.multilevel.multilevel_solver.level())
   # levels.append(pyamg.multilevel.multilevel_solver.level())
   #
   # # store first level data
   # levels[0].A = A
   # levels[0].C = C
   # levels[0].splitting = splitting
   # levels[0].P = P
   # levels[0].R = R
   #
   # # store second level data
   # levels[1].A = R * A * P    # coarse-level matrix
   #
   # pyamg_solver = pyamg.multilevel.multilevel_solver(levels, coarse_solver='splu')
   #
   # # pyamg_solver = pyamg.aggregation.smoothed_aggregation_solver(A)
   # # pyamg_solver = pyamg.aggregation.energy_prolongation_smoother(jacobian)
   # pyamg_solver = pyamg.aggregation.rootnode_solver(A)
   #
   # preconditioner = pyamg_solver.aspreconditioner(cycle='V')
   # solve_pyamg_gmres(pyamg.krylov.fgmres, A, initial_rhs, inv_precond=preconditioner)

   # solve_gmres_diag_precond(jacobian, initial_rhs)
   # solve_gmres_diag_precond(jacobian, initial_rhs, 2)
   # solve_gmres_diag_precond(jacobian, initial_rhs, 3)
   # solve_gmres_diag_precond(jacobian, initial_rhs, 4)
   # solve_gmres_ilu_precond(jacobian, initial_rhs)
   sol_lgmres = solve_lgmres(jacobian, initial_rhs)
   # solve_lgmres_diag_precond(jacobian, initial_rhs)
   # solve_lgmres_diag_precond(jacobian, initial_rhs, 2)
   # solve_lgmres_diag_precond(jacobian, initial_rhs, 3)
   # solve_lgmres_diag_precond(jacobian, initial_rhs, 4)
   # solve_lgmres_ilu_precond(jacobian, initial_rhs)

   gef_sci_diff = sol_scipy - sol_gef
   norm = scipy.linalg.norm(gef_sci_diff)
   print(f'Sci-GEF diff:    {norm:.3e}, rel {norm/scipy.linalg.norm(sol_scipy):.3e}')
   print(f'Sci-AMG diff:    {scipy.linalg.norm(sol_scipy - sol_amg):.3e}')
   print(f'Sci-Lgmres diff: {scipy.linalg.norm(sol_scipy - sol_lgmres):.3e}')


if __name__ == '__main__':
   np.set_printoptions(precision=2)

   parser = argparse.ArgumentParser(description='''Compute a preconditioner for a given input matrix''')
   parser.add_argument('case_name', help='Name of the .npy file where the matrix is stored')
   parser.add_argument('--config', default=f'{script_dir}/test_case.ini')

   main(parser.parse_args())
