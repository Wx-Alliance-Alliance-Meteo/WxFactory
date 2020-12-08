#!/usr/bin/env python

import argparse
import numpy as np
import os
from time import time

import scipy
import scipy.sparse.linalg
from scipy.sparse import load_npz

script_dir = os.path.dirname(os.path.abspath(__file__))

num_iter = 0

def increment_iter():
   global num_iter
   num_iter = num_iter + 1

def reset_iter():
   global num_iter
   num_iter = 0


def load_matrix(case_name):
   print(f'Reading matrix for "{case_name}"')
   case_dir = f'{script_dir}/jacobian/{case_name}'
   os.makedirs(case_dir, exist_ok = True)

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


def solve_gmres(jacobian, rhs, inv_precond = None):

   reset_iter()
   gmres_callback = lambda x: increment_iter()

   t0 = time()
   sol, return_val = scipy.sparse.linalg.gmres(
      jacobian, rhs,
      M = inv_precond,
      # restart = 20,
      restart = 2,
      tol = 1e-7,
      maxiter = 200,
      callback = gmres_callback)
   t1 = time()
   time_ms = (t1 - t0) * 1000

   # print(f'solution: {sol}')
   if return_val != 0:
      print(f'return val: {return_val}')
   print(f'num iterations: {num_iter}, time: {time_ms: 5.0f} ms')

   Ax = jacobian @ sol
   abs_error = scipy.linalg.norm(rhs - Ax)
   rel_error = abs_error / scipy.linalg.norm(rhs)
   # print(f'relative error: {rel_error} (abs {abs_error})')


def solve_lgmres(jacobian, rhs, inv_precond = None):
   reset_iter()
   lgmres_callback = lambda x: increment_iter()

   inner_m = 15

   t0 = time()
   sol, return_val = scipy.sparse.linalg.lgmres(
      jacobian, rhs,
      M = inv_precond,
      tol = 1e-7,
      atol = 1e-8,
      inner_m = inner_m,
      maxiter = 200,
      callback = lgmres_callback
   )
   t1 = time()
   time_ms = (t1 - t0) * 1000

   if return_val != 0:
      print(f'return val: {return_val}')
   print(f'num LGMRES iterations: {(num_iter - 1) * inner_m} - {num_iter * inner_m}, '
         f'time: {time_ms: 5.0f} ms')


def get_diag(matrix, num_stages = 1):
   diagonal_mat = scipy.sparse.diags([matrix[i,i] for i in range(matrix.shape[0])], format = 'csc')

   for d in range(1, num_stages):
      diagonal_mat =\
         diagonal_mat + \
         scipy.sparse.diags([matrix[i+d, i] for i in range(matrix.shape[0] - d)], offsets = -d, format = 'csc') + \
         scipy.sparse.diags([matrix[i, i+d] for i in range(matrix.shape[0] - d)], offsets = d, format='csc')

   return diagonal_mat


def solve_gmres_diag_precond(jacobian, rhs, num_stages = 1):
   print(f'Using a diagonal preconditioner with {(num_stages - 1) * 2 + 1} diagonals')

   diagonal_mat = get_diag(jacobian, num_stages)
   diag_inv = scipy.sparse.linalg.inv(diagonal_mat)

   # print(f'diag mat: {diagonal_mat.toarray()}')
   solve_gmres(jacobian, rhs, inv_precond=diag_inv)


def solve_lgmres_diag_precond(jacobian, rhs, num_stages = 1):
   print(f'Using a diagonal preconditioner with {(num_stages - 1) * 2 + 1} diagonals')

   diagonal_mat = get_diag(jacobian, num_stages)
   diag_inv = scipy.sparse.linalg.inv(diagonal_mat)

   # print(f'diag mat: {diagonal_mat.toarray()}')
   solve_lgmres(jacobian, rhs, inv_precond=diag_inv)


def get_ilu_inverse(jacobian):
   inv_approx_lu = scipy.sparse.linalg.spilu(jacobian, drop_tol = 1e-5, fill_factor = 15)
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
   solve_gmres(jacobian, rhs, inv_precond= inv_approx)


def solve_lgmres_ilu_precond(jacobian, rhs):
   print('Using an ILU preconditioner')
   inv_approx = get_ilu_inverse(jacobian)
   solve_lgmres(jacobian, rhs, inv_precond= inv_approx)


def main():
   parser = argparse.ArgumentParser(description = '''Compute a preconditioner for a given input matrix''')
   parser.add_argument('case_name', help = 'Name of the .npy file where the matrix is stored')
   parser.add_argument('--config', default= f'{script_dir}/test_case.ini')

   args = parser.parse_args()

   jacobian, initial_rhs = load_matrix(args.case_name)
   solve_gmres(jacobian, initial_rhs)
   solve_gmres_diag_precond(jacobian, initial_rhs)
   solve_gmres_diag_precond(jacobian, initial_rhs, 2)
   solve_gmres_diag_precond(jacobian, initial_rhs, 3)
   solve_gmres_diag_precond(jacobian, initial_rhs, 4)
   solve_gmres_ilu_precond(jacobian, initial_rhs)
   solve_lgmres(jacobian, initial_rhs)
   solve_lgmres_diag_precond(jacobian, initial_rhs)
   solve_lgmres_diag_precond(jacobian, initial_rhs, 2)
   solve_lgmres_diag_precond(jacobian, initial_rhs, 3)
   solve_lgmres_diag_precond(jacobian, initial_rhs, 4)
   solve_lgmres_ilu_precond(jacobian, initial_rhs)


if __name__ == '__main__':
   np.set_printoptions(precision = 2)
   main()
