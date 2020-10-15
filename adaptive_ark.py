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

def solve(fe,fi,Ji,tvals,Y0,Be,Bi,rtol,atol,hmin,hmax,nn,p_phi,mu):

   """
   Usage: tvals,Y,nsteps,ierr = solve(fe,fi,Ji,tvals,Y0,Be,Bi,rtol,atol,hmin,hmax,nn,p_phi,mu)

   Adaptive time step, additive Runge-Kutta solver for the vector-valued ODE problem
      y' = fe(Y(t)) + fi(Y(t)), t in tvals, y in R^m,
      Y(t0) = [y1(t0), y2(t0), ..., ym(t0)]'.

   Inputs:
      fe     = function handle for evaluating fe(Y)
      fi     = function handle for evaluating fi(Y) -- assumed linear in Y, i.e., fi(Y)=Ji*Y
      Ji     = function handle for evaluating fi'(Y) -- assumed a fixed matrix
      tvals  = [t0, t1, t2, ..., tN]
      Y0     = initial value array (column vector of length m)
      Be     = Butcher table dictionary for ERK method, containing:
                 'A' = s-by-s matrix of coefficients (strictly lower-triangular)
                 'b' = s array of solution coefficients
                 'c' = s array of abcissae
                 'p' = embedding order
      Bi     = Butcher table dictionary for DIRK method (same fields as Be)
      rtol   = desired relative error of solution  (scalar)
      atol   = desired absolute error of solution  (vector or scalar)
      hmin   = minimum internal time step size (hmin <= t(i)-t(i-1), for all i)
      hmax   = maximum internal time step size (hmax >= hmin)
      nn     = the size of the original problem
      p_phi  = the highest index of the ``φ`` functions
      mu     = normalization factor

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

   if Be['q'] != Bi['q']:
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

   if adaptive:
      de = Be['b2']
      di = Bi['b2']
      if (Be['p'] != Bi['p']):
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
   Ynew = Y0.copy()

   # set initial time step size
   h = hmin

   # initialize work counters
   nsteps = 0

   fgmres_solver = linsol.Fgmres(tol = lin_tol)

   # iterate over output time steps
   for tstep in range(1,len(tvals)):

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

            # perform single Newton update for implicit solve of new stage solution,
            # with initial Newton guess = y_n:
            #     zi = Y0 - A^{-1}*Res(Y0),
            # where
            #     Res(z) = z - h*AI(i,i)*fi(z) - Y0 - h*sum_{j=1}^{i-1} (Ai_{i,j}*fI_j + Ae_{i,j}*fE_j),
            #     A = I - h*AI(i,i)*Ji
            # Simplifying:
            #     zi = Y0 + A^{-1}*rhs, where
            #     rhs = h*[AI(i,i)*fi(Y0) + sum_{j=1}^{i-1} (Ai_{i,j}*fI_j + Ae_{i,j}*fE_j)]
            rhs = h*Ai[stage,stage]*fi(Y0)
            for j in range(stage):
               rhs += h*( Ae[stage,j]*ke[:,j] + Ai[stage,j]*ki[:,j] )

            # if stage requires a solve, then do that here, and set a flag 'lierr'
            # to equal 0 if this solve succeeded, or 1 otherwise
            if (numpy.abs(Ai[stage,stage]) > 1.e-14):
               matvec_gmres = lambda vec: I_minus_tauJ_imp(vec, h*Ai[stage,stage], nn, Ji)
               dz, gmres_error, nb_gmres_iter, lierr = fgmres_solver.solve(matvec_gmres, rhs)

               # if linear solver failed to converge, set relevant flags/statistics
               # and break out of stage loop
               if (lierr != 0):
                  st_fail = 1
                  break
            else:
               dz = rhs

            # construct stage solution
            z = Y0 + dz
               
            # store implicit and explicit RHS at current stage solution
            ke[:,stage] = fe(z)
            ki[:,stage] = fi(z)

         # increment number of internal time steps taken
         nsteps += 1

         # compute new solution and embedding
         Ynew = Y0.copy()
         Yerr = numpy.zeros(m, dtype=float)
         for j in range(s):
            Ynew += h*(ke[:,j]*be[j] + ki[:,j]*bi[j])
            Yerr += h*(ke[:,j]*(be[j]-de[j]) + ki[:,j]*(bi[j]-di[j]))
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
            Y0 = Ynew.copy()
            t += h

            # use error estimate to adapt the time step (I-controller)
            h_old = h
            h = min(h_safety * h_old * err_step**(-1.0/p), h_old*h_growth)

      # Update the last part of Ynew using analytical formula
      for k in range(p_phi-1):
         i = p_phi - k + 1
         Ynew[nn+k] = (tvals[tstep]**i) / math.factorial(i) * mu
      Ynew[nn+p_phi-1] = mu

      # store updated solution in output array
      Y[:,tstep] = Ynew

   return [tvals, Y, nsteps, ierr]


def I_minus_tauJ_imp(vec, hAi, n, J_imp):
   mv = numpy.zeros_like(vec)
   mv[:n] = J_imp(vec[:n])
   return vec - hAi * mv
