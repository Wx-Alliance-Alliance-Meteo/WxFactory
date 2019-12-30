import FastGaussQuadrature

include("constants.jl")
include("cartesian.jl")

struct Geom
   solutionPoints
   extension
   x1
   x2
   Δx1
   Δx2
   X
   Y
   cartX
   cartY
   cartZ
   lon
   lat
end

function cubed_sphere(nb_elements, degree)

   domain_x1 = (-pi/4, pi/4)
   domain_x2 = (-pi/4, pi/4)
   nb_elements_x1 = nb_elements
   nb_elements_x2 = nb_elements

   # Gauss-Legendre solution points
   (solutionPoints, weights) = FastGaussQuadrature.gausslegendre(degree+1)

   nb_solpts = length(solutionPoints)

   extension = [-1 solutionPoints' 1]

   scaled_points = 0.5 * (1.0 .+ solutionPoints)

   # Equiangular coordinates
   Δx1 = (domain_x1[2] - domain_x1[1]) / nb_elements_x1
   Δx2 = (domain_x2[2] - domain_x2[1]) / nb_elements_x2

   faces_x1 = LinRange(domain_x1[1], domain_x1[2], nb_elements_x1 + 1) |> collect
   faces_x2 = LinRange(domain_x2[1], domain_x2[2], nb_elements_x2 + 1) |> collect

   ni = nb_elements_x1 * length(solutionPoints)
   x1 = zeros(ni)
   for i in 1:nb_elements_x1
      idx1 = (i-1) * nb_solpts + 1
      x1[idx1 : idx1 + nb_solpts - 1] = faces_x1[i] .+ scaled_points .* Δx1
   end

   nj = nb_elements_x2 * length(solutionPoints)
   x2 = zeros(nj)
   for i in 1:nb_elements_x2
      idx2 = (i-1) * nb_solpts + 1
      x2[idx2 : idx2 + nb_solpts - 1] = faces_x2[i] .+ scaled_points .* Δx2
   end

   X1 = repeat(reshape(x1, 1, :), length(x2), 1)
   X2 = repeat(x2, 1, length(x1))

   # Gnomonic coordinates
   X = tan.(X1)
   Y = tan.(X2)

   # Cartesian coordinates on unit sphere
   cartX = zeros(ni,nj,nbfaces)
   cartY = zeros(ni,nj,nbfaces)
   cartZ = zeros(ni,nj,nbfaces)
   
   R = 1.0
   a = R / sqrt(3)
   x = a * tan.(X1)
   y = a * tan.(X2)
   proj = R ./ sqrt.(a^2 .+ x.^2 + y.^2)

   cartX[:,:,1] .= proj .* a
   cartY[:,:,1] .= proj .* x
   cartZ[:,:,1] .= proj .* y

   cartX[:,:,2] .= proj .* -x
   cartY[:,:,2] .= proj .* a
   cartZ[:,:,2] .= proj .* y

   cartX[:,:,3] .= proj .* -a
   cartY[:,:,3] .= proj .* -x
   cartZ[:,:,3] .= proj .* y

   cartX[:,:,4] .= proj .* x
   cartY[:,:,4] .= proj .* -a
   cartZ[:,:,4] .= proj .* y

   cartX[:,:,5] .= proj .* -y
   cartY[:,:,5] .= proj .* x
   cartZ[:,:,5] .= proj .* a
   
   cartX[:,:,6] .= proj .* y
   cartY[:,:,6] .= proj .* x
   cartZ[:,:,6] .= proj .* -a
   
   # Spherical coordinates 
   lon = zeros(ni,nj,nbfaces)
   lat = zeros(ni,nj,nbfaces)

   for pannel in 1:4
      lon[:,:,pannel] .= X1 .+ pi/2.0 .* (pannel - 1)
      lat[:,:,pannel] .= atan.(Y .* cos.(X1))
   end

   lon[:,:,5:6], lat[:,:,5:6],  = cart2sph(cartX[:,:,5:6], cartY[:,:,5:6], cartZ[:,:,5:6])
 
   lon[lon.<0] .= lon[lon.<0] .+ (2*pi)

   return Geom(solutionPoints, extension, X1, X2, Δx1, Δx2, X, Y, cartX, cartY, cartZ, lon, lat)

end
