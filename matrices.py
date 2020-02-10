import numpy

class Ops:
   def __init__(self, extrap_west, extrap_east, extrap_south, extrap_north, diff_ext, diff_solpt, correction):
      self.extrap_west  = extrap_west
      self.extrap_east  = extrap_east
      self.extrap_south = extrap_south
      self.extrap_north = extrap_north
      self.diff_ext = diff_ext
      self.diff_solpt = diff_solpt
      self.correction = correction

def set_operators(grd):
   extrap_west = lagrangeEval(grd.solutionPoints, -1)
   extrap_east = lagrangeEval(grd.solutionPoints,  1)

   extrap_south = lagrangeEval(grd.solutionPoints, -1)
   extrap_north = lagrangeEval(grd.solutionPoints,  1)

   diff_ext = diffmat(grd.extension)
   diff_solpt = diff_ext[1:-1, 1:-1]

   correction = numpy.column_stack((diff_ext[1:-1,0], diff_ext[1:-1,-1]))

   return Ops(extrap_west, extrap_east, extrap_south, extrap_north, diff_ext, diff_solpt, correction)

def lagrangeEval(points, pos):
   x = pos
   l = numpy.zeros_like(points)
   for i in range(len(points)):
      l[i] = x / x
      for j in range(len(points)):
         if(i != j):
            l[i] = l[i] * (x-points[j]) / (points[i] - points[j])
   return l

def diffmat(X):
   M = len(X)
   D = numpy.zeros((M,M))
   for i in range(M):
      for j in range(M):
         D[i,j] = dLagrange(j, X[i], X)
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
