import YAML

struct Param
   case_number
   dt
   t_end
   time_integrator
   krylov_size
   tolerance
   α
   degree
   nb_elements
   stat_freq
   plot_freq
end

function get_parameters(argv)

   path = argv[1]
   data = YAML.load(open(path))

   key = "Test case"
   case_number  = data[key]["case_number"]

   key = "Time integration"
   dt              = data[key]["dt"]
   t_end           = data[key]["t_end"]
   time_integrator = data[key]["time_integrator"]
   krylov_size     = data[key]["krylov_size"]
   tolerance       = data[key]["tolerance"]

   key = "Spatial discretization"
   α             = data[key]["α"]
   degree        = data[key]["degree"]
   nb_elements   = data[key]["nb_elements"]

   key = "Plot options"
   stat_freq = data[key]["stat_freq"]
   plot_freq = data[key]["plot_freq"]

   println("\nLoading config: ", path)
   display(data)
   println(" ")

   return Param( case_number,
                 dt, t_end, time_integrator, krylov_size, tolerance,
                 α, degree, nb_elements,
                 stat_freq, plot_freq )

end
