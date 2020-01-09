using Plots
pyplot()

function plot_field(geom, field)

   for pannel in 1:6
      display( surface!(geom.cartX[:,:,pannel], geom.cartY[:,:,pannel], geom.cartZ[:,:,pannel], fill_z=field[:,:,pannel]) )
   end 

end
