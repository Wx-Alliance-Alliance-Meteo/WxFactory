import numpy

class Ops:
   def __init__(self, lcoef, rcoef, diff_ext, diff_solpt, correction):
      self.lcoef = lcoef
      self.rcoef = rcoef
      self.diff_ext = diff_ext
      self.diff_solpt = diff_solpt
      self.correction = correction

def set_operators(grd):
   lcoef = lagrangeEval(grd.solutionPoints, -1)
   rcoef = lagrangeEval(grd.solutionPoints,  1)

   diff_ext = diffmat(grd.extension)
   diff_solpt = diff_ext[1:-1, 1:-1]

   correction = numpy.column_stack((diff_ext[1:-1,0], diff_ext[1:-1,-1]))

   return Ops(lcoef, rcoef, diff_ext, diff_solpt, correction)

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
