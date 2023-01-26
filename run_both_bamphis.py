#!/usr/bin/env python3

import numpy
from oct2py import octave
from scipy.sparse import load_npz
from time import time

from bamphi import bamphi

system_matrix_file = "system_matrix.npz"
rhs_file           = "rhs.out.npy"
try:
   A = load_npz(system_matrix_file)
   u = numpy.load(rhs_file)

   print(f'Loaded system matrix + u from files ({system_matrix_file}, {rhs_file})')
except:
   print(f'Unable to load problem from files ({system_matrix_file} and {rhs_file}). Using default problem')
   raise

def Ax(x):
   return A @ x

def main():
   octave.addpath('/home/vma000/site5/bamphi')
   # octave.run_bamphi()
   # return

   # t = numpy.array([0.25, 0.65, 1.0])
   # t = numpy.linspace(0.1, 1, 5)
   t = numpy.array([1.0])
   print(f't = {t}')

   # def Ax(x):
   #    return A @ x

   # def ATx(x):
   #    return A.T @ x

   # f, info = octave.bamphi(t, lambda x: Ax(x), numpy.empty(0), u)
   t0 = time()
   f_oct   = octave.bamphi_wrapper(t, A, u.T)
   t_oct = time() - t0
   t0 = time()
   f_py,_  = bamphi(t, Ax, u)
   t_py  = time() - t0

   print(f'f_oct = \n{f_oct[:10,-3:]}')
   print(f'f_py  = \n{f_py[-3:,:10].T}')

   diff = numpy.linalg.norm(f_py[1:, :] - f_oct.T) / numpy.linalg.norm(f_oct.T)
   last_diff = numpy.linalg.norm(f_py[-1, :] - f_oct.T[-1, :]) / numpy.linalg.norm(f_oct[-1, :])
   print(f'Diff: {diff:.3e} (last entry {last_diff:.3e})')

   print(f'Times \n Oct: {t_oct:.3f}s \n Py:  {t_py:.3f}s')

if __name__ == '__main__':
   main()
