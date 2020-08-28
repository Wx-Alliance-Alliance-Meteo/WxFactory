import numpy

class DFR_operators:
   def __init__(self, grd):
      self.extrap_west = lagrangeEval(grd.solutionPoints, -1)
      self.extrap_east = lagrangeEval(grd.solutionPoints,  1)

      self.extrap_south = lagrangeEval(grd.solutionPoints, -1)
      self.extrap_north = lagrangeEval(grd.solutionPoints,  1)

      self.diff_ext = diffmat(grd.extension)
      self.diff_solpt = self.diff_ext[1:-1, 1:-1]

      self.correction = numpy.column_stack((self.diff_ext[1:-1,0], self.diff_ext[1:-1,-1]))

      self.diff_solpt_tr = self.diff_solpt.T
      self.correction_tr = self.correction.T

      # Ordinairy differentiation matrices
      self.diff = diffmat(grd.solutionPoints)
      self.diff_tr = self.diff.T

      self.quad_weights = numpy.outer(grd.glweights, grd.glweights)

def lagrangeEval(points, x):
   l = numpy.zeros_like(points)
   for i in range(len(points)):
      l[i] = 1.0
      for j in range(len(points)):
         if(i != j):
            l[i] = l[i] * (x-points[j]) / (points[i] - points[j])
   return l

def diffmat(points):
   M = len(points)
   D = numpy.zeros((M,M))
   for i in range(M):
      dsum = 0.
      for j in range(M):
         if i != j:
            D[i,j] = dLagrange(j, points[i], points)
            dsum += D[i,j]
      D[i, i] = -dsum
   return D

def dLagrange(j, xi, x):
   y = 0
   n = len(x)
   for l in range(n):
      if (l!=j):
         k = 1 / (x[j] - x[l])
         for m in range(n):
            if (m!=j) and (m!=l):
               k = k*(xi-x[m])/(x[j]-x[m])
         y = y + k
   return y
