struct Ops
   lcoef
   rcoef
   diff_ext
   diff_solpt
   correction
end

function matrices(grd)
   lcoef   = lagrangeEval(grd.solutionPoints, -1)
   rcoef   = lagrangeEval(grd.solutionPoints,  1)

   diff_ext = diffmat(grd.extension)
   diff_solpt = diff_ext[2:end-1, 2:end-1]
   correction = [ diff_ext[2:end-1,1] diff_ext[2:end-1,end]]

   return Ops(lcoef, rcoef, diff_ext, diff_solpt, correction)
end

function lagrangeEval(points, pos)
   x = pos
   l = zeros(length(points))
   for i in 1:length(points)
      l[i] = x / x
      for j in 1:length(points)
         if(i != j)
            l[i] = l[i] * (x-points[j]) / (points[i] - points[j])
         end
      end
   end
   return l
end

function diffmat(X)
   M = length(X)
   D = zeros(M,M)
   for i in 1:M
      for j in 1:M
         D[i,j] = dLagrange(j, X[i], X)
      end
   end
   return D
end

function dLagrange(j, xi, x)
   y = 0;
   n = length(x);
   for l in 1:n
      if (l!=j)
         k = 1 / (x[j] - x[l]);
         for m in 1:n
            if (m!=j) && (m!=l)
               k = k*(xi-x[m])/(x[j]-x[m]);
            end
         end
         y = y + k;
      end
   end
   return y
end
