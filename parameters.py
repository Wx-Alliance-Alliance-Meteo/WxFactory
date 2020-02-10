from configparser import ConfigParser

class Param:
   def __init__(self, case_number, Williamson_angle, dt, t_end, time_integrator, krylov_size, tolerance, α, nbsolpts, nb_elements, stat_freq, plot_freq):
      self.case_number      = case_number
      self.Williamson_angle = Williamson_angle
      self.dt               = dt
      self.t_end            = t_end
      self.time_integrator  = time_integrator
      self.krylov_size      = krylov_size
      self.tolerance        = tolerance
      self.α                = α
      self.nbsolpts           = nbsolpts
      self.nb_elements      = nb_elements
      self.stat_freq        = stat_freq
      self.plot_freq        = plot_freq

def get_parameters(cfg_file):

   parser = ConfigParser()
   parser.read(cfg_file)

   print('\nLoading config: ' + cfg_file)
   print(parser._sections)
   print(' ')

   case_number      = parser.getint('Test_case', 'case_number')
   Williamson_angle = parser.getfloat('Test_case', 'Williamson_angle')

   dt               = parser.getfloat('Time_integration', 'dt')
   t_end            = parser.getint('Time_integration', 't_end')
   time_integrator  = parser.get('Time_integration', 'time_integrator')
   krylov_size      = parser.getint('Time_integration', 'krylov_size')
   tolerance        = parser.getfloat('Time_integration', 'tolerance')

   α                = parser.getfloat('Spatial_discretization', 'α')
   nbsolpts           = parser.getint('Spatial_discretization', 'nbsolpts')
   nb_elements      = parser.getint('Spatial_discretization', 'nb_elements')

   stat_freq        = parser.getint('Plot_options', 'stat_freq')
   plot_freq        = parser.getint('Plot_options', 'plot_freq')


   return Param(case_number, Williamson_angle, dt, t_end, time_integrator, krylov_size, \
                tolerance, α, nbsolpts, nb_elements, stat_freq, plot_freq)

