include("constants.jl")

function initialize(geom, case_number)

#   bubble_θ = 303.15
#
   # Initial state at rest, isentropic, hydrostatic
#   nk, ni = size(geom.X)
#   Q = zeros(nk, ni, nb_equations)
#   U = zeros(size(geom.X))
#   W = zeros(size(geom.X))
#   Π = zeros(size(geom.X))
#   θ = ones(size(geom.X)) * bubble_θ
#
#   if case_number == 1
#
#         xc = 500.0
#         zc = 260.0
#         rad = 250.0
#
#         pert = 0.5
#
#         for k in 1:nk
#            for i in 1:ni
#               r = (geom.X[k,i]-xc)^2 + (geom.Z[k,i]-zc)^2
#               if r < rad^2
#                  θ[k,i] = θ[k,i] + pert
#               end
#            end
#         end
#
#
#   elseif case_number == 2
#
#         A = 0.5
#         a = 50.
#         s = 100.
#         x0 = 500.
#         z0 = 260.
#
#         for k in 1:nk
#            for i in 1:ni
#               r = sqrt( (geom.X[k,i]-x0)^2 + (geom.Z[k,i]-z0)^2 )
#               if r <= a
#                  θ[k,i] = θ[k,i] + A
#               else
#                  θ[k,i] = θ[k,i] + A*exp(-((r-a)/s)^2)
#               end
#            end
#         end
#
#   else
#
#         A = 0.5
#         a = 150.
#         s = 50.
#         x0 = 500.
#         z0 = 300.
#
#         for k in 1:nk
#            for i in 1:ni
#               r = sqrt( (geom.X[k,i]-x0)^2 + (geom.Z[k,i]-z0)^2 )
#               if r <= a
#                  θ[k,i] = θ[k,i] + A
#               else
#                  θ[k,i] = θ[k,i] + A*exp(-((r-a)/s)^2)
#               end
#            end
#         end
#
#         A = -0.15
#         a = 0.
#         s = 50.
#         x0 = 560.
#         z0 = 640.
#
#         for k in 1:nk
#            for i in 1:ni
#               r = sqrt( (geom.X[k,i]-x0)^2 + (geom.Z[k,i]-z0)^2 )
#               if r <= a
#                  θ[k,i] = θ[k,i] + A
#               else
#                  θ[k,i] = θ[k,i] + A*exp(-((r-a)/s)^2)
#               end
#            end
#         end
#
#   end
#
#   for k in 1:nk
#      for i in 1:ni
#         Π[k,i] = ( 1.0 - gravity / (cpd * bubble_θ) * geom.Z[k,i])
#      end
#   end
#
#   ρ = P0 ./ (Rd .* bubble_θ) .* Π.^(cvd / Rd)
#
#   Q[:,:,RHO]       = ρ
#   Q[:,:,RHO_U]     = ρ .* U
#   Q[:,:,RHO_W]     = ρ .* W
#   Q[:,:,RHO_THETA] = ρ .* θ
#
#   return Q
return Any

end
