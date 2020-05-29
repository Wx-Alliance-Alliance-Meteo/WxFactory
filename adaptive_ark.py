# adaptive_ark.py
#
# Daniel R. Reynolds
# Department of Mathematics
# Southern Methodist University
# May 2020

import linsol
import math
import numpy
import mpi4py.MPI

def solve(fe,fi,Ji,tvals,Y0,Be,Bi,rtol,atol,hmin,hmax,nn):

   """
   Usage: tvals,Y,nsteps,ierr = adaptive_ark(fe,fi,tvals,Y0,Ae,be,ce,de,Ai,bi,ci,di,p,rtol,atol,hmin,hmax)

   Adaptive time step, additive Runge-Kutta solver for the vector-valued ODE problem
      y' = fe(Y(t)) + fi(Y(t)), t in tvals, y in R^m,
      Y(t0) = [y1(t0), y2(t0), ..., ym(t0)]'.

   Inputs:
      fe     = function handle for evaluating fe(Y)
      fi     = function handle for evaluating fi(Y)
      tvals  = [t0, t1, t2, ..., tN]
      Y0     = initial value array (column vector of length m)
      Ae,be,ce,de = Butcher table matrices for ERK method:
         Ae is an s-by-s matrix of coefficients (strictly lower-triangular)
         be is an array of s solution coefficients
         ce is an array of s abcissae for each stage
         de is an array of s embedding coefficients
      Ai,bi,ci,di = Butcher table matrices for DIRK method
      p      = order of accuracy for embedded solution
      rtol   = desired relative error of solution  (scalar)
      atol   = desired absolute error of solution  (vector or scalar)
      hmin   = minimum internal time step size (hmin <= t(i)-t(i-1), for all i)
      hmax   = maximum internal time step size (hmax >= hmin)
      nn     = lenght of the "useful" part of the solution

   Outputs:
      tvals  = the same as the input array tvals
      y      = [y(t0), y(t1), y(t2), ..., y(tN)], where each
                y(t*) is a column vector of length m.
      nsteps = number of internal time steps taken by method
      ierr   = flag denoting success (0) or failure (1)

   Assumptions:
   (1) fi depends linearly on Y, so each implicit stage requires only a linear solve (and not Newton).
   (2) The Butcher tables (Ae,be,ce,de) and (Ai,bi,ci,di) have an identical number of stages.
   (3) Ae is strictly lower-triangular.
   (4) Ai is lower-triangular.
   (5) The ARK method order is >= the embedding order, p.

   Note: we do not assume that ci equals ce.
   """

   # check for compatible tables
   if Be['A'].shape != Bi['A'].shape:
      print('ark.solve error: Be and Bi must have the same number of stages')
      exit(1)

   # number of stages
   s = Be['A'].shape[0]

   if Be['p'] != Bi['p']:
      print('ark.solve error: Be and Bi must have the same method order')

   # extract ERK method information from Be
   ce = Be['c']             # stage time fraction array
   be = Be['b']             # solution weights
   Ae = Be['A']             # RK coefficients
   de = be.copy()           # embedding coefficients (may be overwritten)

   # extract DIRK method information from Bi
   ci = Bi['c']             # stage time fraction array
   bi = Bi['b']             # solution weights
   Ai = Bi['A']             # RK coefficients
   di = bi.copy()           # embedding coefficients (may be overwritten)

   # if adaptivity desired, check for embedding coefficients and set the
   # order of accuracy accordingly
   p = Be['p']

   eps = numpy.finfo(float).eps

   # check whether adaptivity is desired
   adaptive = (abs(hmax-hmin)/abs(hmax) > math.sqrt(eps))

   if adaptive is True:
      de = Be['b2']
      di = Bi['b2']
      if not numpy.array_equal(de,di):
         print('solve_ARK error: Be and Bi must have the same embedding order')

   # initialize outputs
   N = len(tvals)
   if (numpy.isscalar(Y0)):
       m = 1
   else:
       m = len(Y0)
   Y = numpy.zeros((m,N), dtype=float)
   Y[:,0] = Y0
   ierr = 0

   # set the solver parameters
   h_cfail    = 0.25        # failed linear solve step reduction factor
   h_reduce   = 0.1         # failed step reduction factor
   h_safety   = 0.96        # adaptivity safety factor
   h_growth   = 10          # adaptivity growth bound
   lin_tol    = 1e-7        # implicit solver tolerance factor
   e_bias     = 1.5         # error bias factor
   ONEMSM     = 1-1e-10     # coefficient to account for floating-point roundoff
   ERRTOL     = 1.1         # upper bound on allowed step error
                            #   (in WRMS norm)

   # initialize temporary variables
   t  = tvals[0]
   z  = numpy.zeros(m, dtype=float)
   ki = numpy.zeros((m,s), dtype=float)
   ke = numpy.zeros((m,s), dtype=float)
   Ynew = Y0.copy() # TODO : requis ???

   # set initial time step size
   h = hmin

   # initialize work counters
   nsteps = 0

   # iterate over output time steps
   for tstep in range(len(tvals)):

      # loop over internal time steps to get to desired output time
      while (t < tvals[tstep]*ONEMSM):

         # bound internal time step
         h = max(h, hmin)            # enforce minimum time step size
         h = min(h, hmax)            # enforce maximum time step size
         h = min(h, tvals[tstep]-t)  # stop at output time

         # set error-weight vector for this step
         ewt = 1/(rtol*numpy.abs(Y0)+atol)

         # reset stage failure flag
         st_fail = 0

         # loop over stages
         for stage in range(s):

            # set 'time' for current [implicit] stage
            tcur = t + h*ci[stage]

            # compute implicit RHS using known data:
            #     zi = y_n + h*sum_{j=1}^i (Ai(i,j)*fI_j) + h*sum_{j=1}^{i-1} (Ae(i,j)*fE_j)
            #  <=>
            #     zi - h*(a(i,i)*fi) = y_n + h*sum_{j=1}^{i-1} (Ai(i,j)*fI_j + Ae(i,j)*fE_j)
            #  =>
            #     rhs = y_n + h*sum_{j=1}^{i-1} (Ai(i,j)*fI_j + Ae(i,j)*fE_j)
            rhs = Y0
            for j in range(stage-1):
               rhs += h*( Ae[stage,j]*ke[:,j] + Ai[stage,j]*ki[:,j])


            # solve implicit linear system for new stage solution,
            #    A*z = rhs
            # where
            #    A = I - h*AI(i,i)*Ji
            #    Ji = \frac{\partial}{\partial Y} fi(tcur,Y0)
            # to a tolerance || A*z - rhs ||_wrms <= lin_tol.
            #
            # set a flag 'lierr' to equal 0 if this solve succeeded, or 1 otherwise
            matvec_gmres = lambda vec: I_minus_tauJ_imp(vec, h*Ai[stage,stage], nn, Ji)
            z, gmres_error, nb_gmres_iter, lierr = linsol.gmres_mgs(matvec_gmres, rhs, tol=lin_tol)

            # if linear solver failed to converge, set relevant flags/statistics
            # and break out of stage loop
            if (lierr != 0):
               st_fail = 1
               break

            # store implicit and explicit RHS at current stage solution
            ke[:,stage] = fe(z)
            ki[:,stage] = fi(z)

         # increment number of internal time steps taken
         nsteps += 1

         # if a stage solve failed
         if (st_fail == 1):

            # if already at minimum step, just return with failure
            if (numpy.abs(h) <= numpy.abs(hmin)):
               print(f'Stage solve failure at minimum step size (t={tcur}).\n  Consider reducing hmin or increasing rtol.\n')
               return [tvals, Y, nsteps, 1]

            # otherwise, reduce time step and retry step
            h *= h_cfail
            continue

         # if we made it to this point, then all stage solves succeeded

         # compute new solution and embedding
         Ynew = Y0
         Yerr = numpy.zeros(m, dtype=float)
         for j in range(s):
            Ynew += h*(ke[:,j]*be[j] + ki[:,j]*bi[j])
            Yerr += h*(ke[:,j]*(be[j]-de[j]) + ki[:,j]*(de[j]-di[j]))

         # estimate error in current step
         local_sum = numpy.sum((Yerr*ewt)**2)/m
         err_step = e_bias * max(numpy.sqrt( mpi4py.MPI.COMM_WORLD.allreduce(local_sum) ), eps)

         # if error too high, either reduce step size or return with failure
         if (err_step > ERRTOL):

            # if already at minimum step, just return with failure
            if (h <= hmin):
               print(f'Cannot achieve desired accuracy at minimum step size (t={tcur}).\n  Consider reducing hmin or increasing rtol.\n')
               return [tvals, Y, nsteps, 1]

            # otherwise, reduce time step and retry step
            h_old = h
            h = min(h_safety * h_old * err_step**(-1.0/p), h_old*h_reduce)

         # otherwise step was successful (solves succeeded, and error acceptable)
         else:

            # update solution and time for last successful step
            Y0 = Ynew
            t += h

            # use error estimate to adapt the time step (I-controller)
            h_old = h
            h = h_safety * h_old * err_step**(-1.0/p)

            # enforce maximum growth rate on step sizes
            h = min(h_growth*h_old, h)

      # store updated solution in output array
      Y[:,tstep] = Ynew

   return [tvals, Y, nsteps, ierr]


def I_minus_tauJ_imp(vec, hAi, n, J_imp):
   mv = numpy.zeros_like(vec)
   mv[:n] = J_imp(vec[:n])
   return vec - hAi * mv
