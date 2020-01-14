include("constants.jl")

function rhs_fun(Q, grd, mtrx, nbsolpts::Int64, nb_elements_x::Int64, nb_elements_z::Int64, α::Float64)

   type_vec = typeof(Q[1,1,1])

   nb_interfaces_x = nb_elements_x - 1
   nb_interfaces_z = nb_elements_z - 1

   flux_x1          = zeros(type_vec, size(Q))
   flux_x3          = zeros(type_vec, size(Q))

   df1_dx           = zeros(type_vec, size(Q))
   df3_dz           = zeros(type_vec, size(Q))
   dQ_dx            = zeros(type_vec, size(Q))
   dQ_dz            = zeros(type_vec, size(Q))
   Q_xx             = zeros(type_vec, size(Q))
   Q_zz             = zeros(type_vec, size(Q))
   upwind_diffusion = zeros(type_vec, size(Q))

   kfaces_flux      = zeros(type_vec, 2, nbsolpts*nb_elements_x, nb_equations, nb_elements_z)
   kfaces_var       = zeros(type_vec, 2, nbsolpts*nb_elements_x, nb_equations, nb_elements_z)
   kfaces_diffusion = zeros(type_vec, 2, nbsolpts*nb_elements_x, nb_equations, nb_elements_z)

   ifaces_flux      = zeros(type_vec, nbsolpts*nb_elements_z, 2, nb_equations, nb_elements_x)
   ifaces_var       = zeros(type_vec, nbsolpts*nb_elements_z, 2, nb_equations, nb_elements_x)
   ifaces_diffusion = zeros(type_vec, nbsolpts*nb_elements_z, 2, nb_equations, nb_elements_x)

   kfaces_pres = zeros(type_vec, 2, nbsolpts*nb_elements_x, nb_elements_z)
   ifaces_pres = zeros(type_vec, nbsolpts*nb_elements_z, 2, nb_elements_x)

   # Unpack physical variables
   uu       = Q[:,:,RHO_U] ./ Q[:,:,RHO]
   ww       = Q[:,:,RHO_W] ./ Q[:,:,RHO]
   pressure = P0 * (Q[:,:,RHO_THETA] * Rd / P0).^(cpd / cvd)

   # Compute the fluxes
   flux_x1[:,:,1] = Q[:,:,RHO_U]
   flux_x1[:,:,2] = Q[:,:,RHO_U] .* uu + pressure
   flux_x1[:,:,3] = Q[:,:,RHO_U] .* ww
   flux_x1[:,:,4] = Q[:,:,RHO_THETA] .* uu

   flux_x3[:,:,1] = Q[:,:,RHO_W]
   flux_x3[:,:,2] = Q[:,:,RHO_W] .* uu
   flux_x3[:,:,3] = Q[:,:,RHO_W] .* ww + pressure
   flux_x3[:,:,4] = Q[:,:,RHO_THETA] .* ww

   # Interpolate to the element interface
   for elem in 1:nb_elements_z
      epais = (elem-1) * (nbsolpts) .+ (1:nbsolpts)

      for eq in 1:nb_equations
         kfaces_flux[1,:,eq,elem] = mtrx.lcoef' * flux_x3[epais,:,eq]
         kfaces_flux[2,:,eq,elem] = mtrx.rcoef' * flux_x3[epais,:,eq]

         kfaces_var[1,:,eq,elem] = mtrx.lcoef' * Q[epais,:,eq]
         kfaces_var[2,:,eq,elem] = mtrx.rcoef' * Q[epais,:,eq]

         kfaces_pres[1,:,elem] = mtrx.lcoef' * pressure[epais,:]
         kfaces_pres[2,:,elem] = mtrx.rcoef' * pressure[epais,:]
      end
   end


   for elem in 1:nb_elements_x
      epais = (elem-1) * (nbsolpts) .+ (1:nbsolpts)

      for eq in 1:nb_equations
         ifaces_flux[:,1,eq,elem] = flux_x1[:,epais,eq] * mtrx.lcoef
         ifaces_flux[:,2,eq,elem] = flux_x1[:,epais,eq] * mtrx.rcoef

         ifaces_var[:,1,eq,elem] = Q[:,epais,eq] * mtrx.lcoef
         ifaces_var[:,2,eq,elem] = Q[:,epais,eq] * mtrx.rcoef

         ifaces_pres[:,1,elem] = pressure[:,epais] * mtrx.lcoef
         ifaces_pres[:,2,elem] = pressure[:,epais] * mtrx.rcoef
      end
   end

   # Bondary treatement
   # Zeros flux BCs everywhere ...
   kfaces_flux[1,:,:,1]   .= 0.
   kfaces_flux[2,:,:,end] .= 0.

   ifaces_flux[:,1,:,1]   .= 0.
   ifaces_flux[:,2,:,end] .= 0.

   kfaces_diffusion[1,:,:,1]   .= 0.
   kfaces_diffusion[2,:,:,end] .= 0.

   ifaces_diffusion[:,1,:,1]   .= 0.
   ifaces_diffusion[:,2,:,end] .= 0.

   # ... except for momentum eqs where pressure is extrapolated to BCs.
   kfaces_flux[1,:,RHO_W,1]   = kfaces_pres[1,:,1]
   kfaces_flux[2,:,RHO_W,end] = kfaces_pres[2,:,end]

   ifaces_flux[:,1,RHO_U,1]   = ifaces_pres[:,1,1]  # TODO : theo seulement ...
   ifaces_flux[:,2,RHO_U,end] = ifaces_pres[:,2,end]

   # Common Rusanov fluxes
   for itf = 1:nb_interfaces_z

      eig_L = abs.(kfaces_var[2,:,RHO_W,itf]   ./ kfaces_var[2,:,RHO,itf])   + sqrt.( heat_capacity_ratio .* kfaces_pres[2,:,itf]   ./ kfaces_var[2,:,RHO,itf]  )
      eig_R = abs.(kfaces_var[1,:,RHO_W,itf+1] ./ kfaces_var[1,:,RHO,itf+1]) + sqrt.( heat_capacity_ratio .* kfaces_pres[1,:,itf+1] ./ kfaces_var[1,:,RHO,itf+1])

      kfaces_flux[1,:,:,itf+1] = 0.5 .* ( kfaces_flux[2,:,:,itf] + kfaces_flux[1,:,:,itf+1] - max(abs.(eig_L), abs.(eig_R)) .* ( kfaces_var[1,:,:,itf+1] - kfaces_var[2,:,:,itf] ) )
      kfaces_flux[2,:,:,itf]   = kfaces_flux[1,:,:,itf+1]

      kfaces_diffusion[1,:,:,itf+1] = 0.5 .* ( kfaces_var[2,:,:,itf] + kfaces_var[1,:,:,itf+1] )
      kfaces_diffusion[2,:,:,itf]   = kfaces_diffusion[1,:,:,itf+1]

   end

   for itf = 1:nb_interfaces_x

      eig_L = abs.(ifaces_var[:,2,RHO_U,itf]   ./ ifaces_var[:,2,RHO,itf])   + sqrt.( heat_capacity_ratio .* ifaces_pres[:,2,itf]   ./ ifaces_var[:,2,RHO,itf] )
      eig_R = abs.(ifaces_var[:,1,RHO_U,itf+1] ./ ifaces_var[:,1,RHO,itf+1]) + sqrt.( heat_capacity_ratio .* ifaces_pres[:,1,itf+1] ./ ifaces_var[:,1,RHO,itf+1] )

      ifaces_flux[:,1,:,itf+1] = 0.5 .* ( ifaces_flux[:,2,:,itf] + ifaces_flux[:,1,:,itf+1] - max(abs.(eig_L), abs.(eig_R)) .* ( ifaces_var[:,1,:,itf+1] - ifaces_var[:,2,:,itf] ) )
      ifaces_flux[:,2,:,itf]   = ifaces_flux[:,1,:,itf+1]

      ifaces_diffusion[:,1,:,itf+1] = 0.5 .* ( ifaces_var[:,2,:,itf] + ifaces_var[:,1,:,itf+1] )
      ifaces_diffusion[:,2,:,itf]   = ifaces_diffusion[:,1,:,itf+1]
   end

   # Compute the derivatives
   for elem in 1:nb_elements_z
      epais = (elem-1) * (nbsolpts) .+ (1:nbsolpts)

      for eq in 1:nb_equations
         df3_dz[epais,:,eq] = ( mtrx.diff_solpt * flux_x3[epais,:,eq] + mtrx.correction * kfaces_flux[:,:,eq,elem] ) * 2.0/grd.Δz
         dQ_dz[epais,:,eq]  = ( mtrx.diff_solpt * Q[epais,:,eq] + mtrx.correction * kfaces_diffusion[:,:,eq,elem] ) * 2.0/grd.Δz
      end
   end

   for elem in 1:nb_elements_x
      epais = (elem-1) * (nbsolpts) .+ (1:nbsolpts)

      for eq in 1:nb_equations
         df1_dx[:,epais,eq] = ( flux_x1[:,epais,eq] * mtrx.diff_solpt' + ifaces_flux[:,:,eq,elem] * mtrx.correction' ) * 2.0/grd.Δx
         dQ_dx[:,epais,eq]  = ( Q[:,epais,eq] * mtrx.diff_solpt' + ifaces_diffusion[:,:,eq,elem] * mtrx.correction' ) * 2.0/grd.Δx
      end
   end

   # Interpolate first derivative of diffusion to the interface
   for elem in 1:nb_elements_z
      epais = (elem-1) * (nbsolpts) .+ (1:nbsolpts)

      for eq in 1:nb_equations
         kfaces_var[1,:,eq,elem] = mtrx.lcoef' * dQ_dz[epais,:,eq]
         kfaces_var[2,:,eq,elem] = mtrx.rcoef' * dQ_dz[epais,:,eq]
      end
   end

   for elem in 1:nb_elements_x
      epais = (elem-1) * (nbsolpts) .+ (1:nbsolpts)

      for eq in 1:nb_equations
         ifaces_var[:,1,eq,elem] = dQ_dx[:,epais,eq] * mtrx.lcoef
         ifaces_var[:,2,eq,elem] = dQ_dx[:,epais,eq] * mtrx.rcoef
      end
   end

   # Communication at cell interfaces with central flux
   for itf in 1:nb_interfaces_z
      kfaces_diffusion[1,:,:,itf+1] = 0.5 .* ( kfaces_var[2,:,:,itf] + kfaces_var[1,:,:,itf+1] )
      kfaces_diffusion[2,:,:,itf]   = kfaces_diffusion[1,:,:,itf+1]
   end

   for itf in 1:nb_interfaces_x
      ifaces_diffusion[:,1,:,itf+1] = 0.5 .* ( ifaces_var[:,2,:,itf] + ifaces_var[:,1,:,itf+1] )
      ifaces_diffusion[:,2,:,itf]   = ifaces_diffusion[:,1,:,itf+1]
   end

   # Bondary treatement (this should be equivalent to a zero diffusion coefficient at the boundary)
   kfaces_diffusion[1,:,:,1]   .= 0.
   kfaces_diffusion[2,:,:,end] .= 0.

   ifaces_diffusion[:,1,:,1]   .= 0.
   ifaces_diffusion[:,2,:,end] .= 0.

   # Finalize the diffusion operator
   for elem in 1:nb_elements_z
      epais = (elem-1) * (nbsolpts) .+ (1:nbsolpts)

      for eq in 1:nb_equations
         Q_zz[epais,:,eq] = ( mtrx.diff_solpt * dQ_dz[epais,:,eq] + mtrx.correction * kfaces_diffusion[:,:,eq,elem] ) * 2.0/grd.Δz
      end
   end

   for elem in 1:nb_elements_x
      epais = (elem-1) * (nbsolpts) .+ (1:nbsolpts)

      for eq in 1:nb_equations
         Q_xx[:,epais,eq] = ( dQ_dx[:,epais,eq] * mtrx.diff_solpt' + ifaces_diffusion[:,:,eq,elem] * mtrx.correction' ) * 2.0/grd.Δx
      end
   end

   for eq in 1:nb_equations
      upwind_diffusion[:,:,eq] = (α * 0.5 * grd.Δx / nbsolpts) .* abs.(uu) .* Q_xx[:,:,eq] .+ (α * 0.5 * grd.Δz / nbsolpts) .* abs.(ww) .* Q_zz[:,:,eq]
   end

   # Assemble the right-hand sides
   rhs = - df1_dx - df3_dz + upwind_diffusion

   rhs[:,:,RHO_W] = rhs[:,:,RHO_W] - Q[:,:,RHO] .* gravity

   return rhs

end
