using Makie

function plot_cubedsphere(geom)

   for pannel in 1:6
      display(surface!(geom.cartX[:,:,pannel], geom.cartY[:,:,pannel], geom.cartZ[:,:,pannel]))
   end 

end
