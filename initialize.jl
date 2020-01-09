include("constants.jl")

function initialize(geom, case_number, α)

   ni, nj, = size(geom.lon)

   if case_number == 1
      # Initialize gaussian bell

      lon_center = 3.0 * pi / 2.0
      lat_center = 0.0

      h0 = 1000.0

      radius = 1.0 / 3.0

      dist = acos.(sin(lat_center) .* sin.(geom.lat) .+ cos(lat_center) .* cos.(geom.lat) .* cos.(geom.lon .- lon_center))

      h = 0.5 .* h0 .* (1.0 .+ cos.(pi .* dist ./ radius)) .* (dist .<= radius)

#      h_analytic = h
#      hsurf = zeros(ni, nj, nbfaces)
   end

   if case_number == 1
      # Solid body rotation

      u¹ = zeros(ni, nj, nbfaces)
      u² = zeros(ni, nj, nbfaces)

      u0 = 2.0 * pi * earth_radius / (12.0 * day_in_secs)
      sinα = sin(α)
      cosα = cos(α)

      u¹[:,:,1] .= u0 ./ earth_radius .* (cosα .+ geom.Y ./ (1.0 .+ geom.X.^2) .* sinα)
      u²[:,:,1] .= u0 .* geom.X ./ (earth_radius .* (1.0 .+ geom.Y.^2)) .* (geom.Y .* cosα .- sinα)

      u¹[:,:,2] .= u0 ./ earth_radius .* (cosα .- geom.X .* geom.Y ./ (1.0 .+ geom.X.^2) .* sinα)
      u²[:,:,2] .= u0 ./ earth_radius .* (geom.X .* geom.Y ./ (1.0 .+ geom.Y.^2) .* cosα .- sinα)

      u¹[:,:,3] .= u0 ./ earth_radius .* (cosα .- geom.Y ./ (1.0 .+ geom.X.^2) .* sinα)
      u²[:,:,3] .= u0 .* geom.X ./ (earth_radius .* (1.0 .+ geom.Y.^2)) .* (geom.Y .* cosα .+ sinα)

      u¹[:,:,4] .= u0 ./ earth_radius .* (cosα .+ geom.X .* geom.Y ./ (1.0 .+ geom.X.^2) .* sinα)
      u²[:,:,4] .= u0 ./ earth_radius .* (geom.X .* geom.Y ./ (1.0 .+ geom.Y.^2) .* cosα .+ sinα)

      u¹[:,:,5] .= u0 ./ earth_radius .* (- geom.Y ./ (1.0 .+ geom.X.^2) .* cosα .+ sinα)
      u²[:,:,5] .= u0 .* geom.X ./ (earth_radius .* (1.0 .+ geom.Y.^2)) .* (cosα .+ geom.Y .* sinα)

      u¹[:,:,6] .= u0 ./ earth_radius .* (geom.Y ./ (1.0 .+ geom.X.^2) .* cosα .- sinα)
      u²[:,:,6] .=-u0 .* geom.X ./ (earth_radius .* (1.0 .+ geom.Y.^2)) .* (cosα .+ geom.Y .* sinα)

#      u = u¹_contra * earth_radius * 2/grd.elementSize
#      v = u²_contra * earth_radius * 2/grd.elementSize

   end

   Q = zeros(ni, nj, nbfaces, nb_equations)

   Q[:,:,:,1] .= h
   Q[:,:,:,2] .= h .* u¹
   Q[:,:,:,3] .= h .* u²

   return Q

end
