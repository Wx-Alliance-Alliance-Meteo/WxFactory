import math
import numpy
import sys

from integrators.butcher import *
from scipy.sparse.linalg import eigs

def exode(τ_out, A, u, method='ARK3(2)4L[2]SA-ERK', controller="deadbeat", rtol=1e-3, atol = 1e-6, task1 = False, verbose=False):
   
   if not hasattr(exode, "first_step"):
      exode.first_step = τ_out # TODO : use CFL condition ?

   # TODO : implement dense output for output at intermediate values of τ_out

   ppo, n = u.shape
   p = ppo - 1
   
   if p == 0:
      p = 1
      # Add extra column of zeros
      u = numpy.row_stack((u, numpy.zeros(len(u))))

   y0 = u[0].copy()

   def fun(t, x):
      ret = A(x)
      for j in range(p):
         ret += t**j/math.factorial(j) * u[j+1]
      return ret

   method = method.upper()

   if method not in METHODS:
      raise ValueError("`method` must be one of {}." .format(METHODS))
   else:
      method = METHODS[method]

   t0, tf = map(float, [0, 1.]) # τ_out])
   
   solver = method(fun, t0, y0, tf, controller=controller, first_step=exode.first_step, rtol=rtol, atol=atol)

   ts = [t0]

   status = None
   while status is None:
      solver.step()

      if solver.status == 'finished':
         status = 0
         
         ''' TODO: put this in a better location. Computing eigenvalues will significantly delay the 
         computation process. Only here for now to test eigenvalue of the 2D ADR problem. 

         # Compute eigenvalues
         matrix_size = y0.shape[0] 
         MATRIX = numpy.zeros((matrix_size,matrix_size)) #numpy.empty([matrix_size,matrix_size])
         for i in range(matrix_size):
             vec = numpy.zeros(matrix_size) 
             vec[i] = 1.
             new_col = A(vec)
             MATRIX[:,i] = new_col 
         
         # Compute eigenvalues
         eigenvalues, eigenvectors = numpy.linalg.eig(MATRIX) #eigs(MATRIX, k=1)
         eigenvalues_real = eigenvalues.real
         eigenvalues_imag = eigenvalues.imag
         
         eig_file = "/home/siw001/gef/vicky/ADR_2D/eigenvalues/ADR_2D_eigenvalues_Nx_401_Nstep_50.csv"
         with open(eig_file, "ab") as foutput:
             foutput.write(b"\n")
             numpy.savetxt(foutput, eigenvalues_real, delimiter=",")
             numpy.savetxt(foutput, eigenvalues_imag, delimiter=",") 
         
         # Calculuate and save the eigenvalues to file. TODO: need better ways
         #for i in range(matrix_size):
            #sys.stdout.write(str(eigenvalues[i].real) + " " )
         #sys.stdout.write("\n")

         #for i in range(matrix_siz):
            #sys.stdout.write(str(eigenvalues[i].imag) + " " )
         #sys.stdout.write("\n")
         
         #print("eigenvalues = ", eigenvalues) 
         #numpy.savetxt('/home/siw001/gef/vicky/ADR_2D/testoutput/grid_size_160000/eigenvalues/ADR_2D_real_eig.csv', eigenvalues_real, delimiter=',')
         #numpy.savetxt('/home/siw001/gef/vicky/ADR_2D/testoutput/grid_size_160000/eigenvalues/ADR_2D_real_eig.csv', eigenvalues_imag, delimiter=',')
         '''

      elif solver.status == 'failed':
         status = -1
         break

      t_old = solver.t_old
      t = solver.t
      y = solver.y

      ts.append(t)

   ts = numpy.array(ts)

   solution = solver.y
   stats = (solver.nfev, solver.failed_steps, solver.error_estimation,solver.error_norm_old, solver.h_previous, solver.h) # TODO
   # keep track of h_previous, use as first step for next iteration. 
   # print("previous step = ", stats[4], "final step = ", stats[5])

   exode.first_step = numpy.median(numpy.diff(ts))

   return solution, stats
