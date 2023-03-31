from integrators.stepper import Stepper
import numpy as np
from scipy import linalg, sparse
from math import sqrt, cos, pi
from scipy.sparse.linalg.dsolve import linsolve
import timeit
from output.solver_stats import write_solver_stats
from solvers.nonlin      import newton_krylov

class explicitRK(Stepper):
   def __init__(self, rhs, RKmethod):
      super().__init__()
      self.rhs = rhs
      self.RKmethod = RKmethod
      print("Use RKmethod in explicitRK.py", self.RKmethod)

   def __step__(self, Q, dt):
      tableau = tableaus[self.RKmethod]
      Q = Tableau.y_step(tableau, self.rhs,Q,dt)
      return Q




class Tableau:
   def __init__(self, c, a, b, explicit=False, sdirk=False):
      self._c=c
      self._a=a
      self._b=b
      self._sizeb=b.size
      self._explicit=explicit
      self._sdirk=sdirk



   def y_step(self, rhs, Q, dt):
      if self._explicit:
         return Q + dt*np.dot(self.get_k_values_explicit(Q,rhs,dt), self._b)
      elif self._sdirk:
         return Q + dt.np.dot(self.get_k_values_sdirk(Q,rhs,dt), self._b)

   def get_k_values_explicit(self, Q, rhs, dt):
      """Return the k values (slopes) for the next step of the solution
         for an explicit Runge--Kutta method"""
      kvalues = np.zeros((Q.shape[0], Q.shape[1], Q.shape[2], self._sizeb), Q.dtype)
      for i in range(np.size(self._b)):
         k_i = rhs(Q + dt * np.dot(kvalues, self._a[i]))
         kvalues[:, :, :, i] = k_i
      return kvalues

   def get_k_values_sdirk(self, Q, rhs, dt):
       """Return the k values (slopes) for the next step of the solution
          for an Singly-Diagonal Implicit Runge--Kutta method (SDIRK) """
       kvalues = np.zeros((Q.shape[0], Q.shape[1], Q.shape[2], self._sizeb), Q.dtype)
       for i in range(self._sizeb):
           #print("i=", i)

           def stagek_system(Q_i, Q, dt, rhs):
               return (Q_i - Q) / dt - np.dot(kvalues, self._a[i])

           #print("enter step")
           stagek_fun = lambda Q_i: stagek_system(Q_i, Q, dt, rhs)

           maxiter = None
           # if self.preconditioner is not None:
           #   self.preconditioner.prepare(dt, Q)
           #   maxiter = 800

           # Update solution
           newQ, nb_iter, residuals = newton_krylov(stagek_fun, Q, f_tol=1e-8, fgmres_restart=30,
                                                           fgmres_precond="none", verbose=False,
                                                           maxiter=maxiter)
           newQ = np.reshape(newQ, Q.shape)
           kvalues[:, :, :, i] = rhs(newQ)
       return kvalues

tableaus = {'fe': Tableau(np.array([0]),np.array([[0]]),np.array([1]), explicit=True),
            'rk4': Tableau(np.array([0, 0.5, 0.5, 1]),
                           np.array([[0, 0, 0, 0],
                                     [0.5, 0, 0, 0],
                                     [0, 0.5, 0, 0],
                                     [0, 0, 1, 0]]),
                           np.array([1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6]), explicit=True),
            'heun': Tableau(np.array([0, 1]),
                            np.array([[0, 0],
                                      [1, 0]]),
                            np.array([0.5, 0.5]),
                            explicit=True),
            'rk3': Tableau(np.array([0, 0.5, 1]),
                           np.array([[0, 0, 0], [0.5, 0, 0], [-1, 2, 0]]),
                           np.array([1. / 6, 2. / 3, 1. / 6]), explicit=True),
            'be': Tableau(np.array([1]), np.array([[1]]), np.array([1]), sdirk=True),
            'cn': Tableau(np.array([0, 1]), np.array([[0, 0], [0.5, 0.5]]),
                          np.array([0.5, 0.5]), sdirk=True),
            'ssp(5,4)': Tableau(np.array([0, 0.39175222700392, 0.58607968896779,
                                          0.47454236302687, 0.93501063100924]),
                                np.array([[0, 0, 0, 0, 0],
                                          [0.39175222700392, 0, 0, 0, 0],
                                          [0.21766909633821, 0.36841059262959,
                                           0, 0, 0],
                                          [0.08269208670950, 0.13995850206999,
                                           0.25189177424738, 0, 0],
                                          [0.06796628370320, 0.11503469844438,
                                           0.20703489864929, 0.54497475021237,
                                           0]]),
                                np.array([0.14681187618661, 0.24848290924556,
                                          0.10425883036650, 0.27443890091960,
                                          0.22600748319395]), explicit=True),
            #          'EXACT': Tableau(np.array([0]), np.array([[0]]), np.array([1]),explicit=True),   ## "tableaus" for using exact value, with all k-entries = 0
            'tvdrk3': Tableau(np.array([0, 1., 0.5]),
                              np.array([[0, 0, 0], [1., 0, 0], [1. / 4, 1. / 4, 0]]),
                              np.array([1. / 6, 1. / 6, 2. / 3]), explicit=True),
            # SSPRK3 is the same as TVDRK3
}

