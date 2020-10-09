from configparser import ConfigParser, NoSectionError, NoOptionError

class Configuration:
   def __init__(self, cfg_file):

      parser = ConfigParser()
      parser.read(cfg_file)

      print('\nLoading config: ' + cfg_file)
      print(parser._sections)
      print(' ')

      self.case_number      = parser.getint('Test_case', 'case_number')
      
      if self.case_number == 9:
         self.matsuno_wave_type = parser.get('Test_case', 'matsuno_wave_type')
         self.matsuno_amp = parser.getfloat('Test_case', 'matsuno_amp')

      self.dt               = parser.getfloat('Time_integration', 'dt')
      self.t_end            = parser.getint('Time_integration', 't_end')
      self.time_integrator  = parser.get('Time_integration', 'time_integrator')
      self.krylov_size      = parser.getint('Time_integration', 'krylov_size')
      self.tolerance        = parser.getfloat('Time_integration', 'tolerance')

      try:
         self.λ0 = parser.getfloat('Grid', 'λ0')
      except (NoOptionError,NoSectionError):
         self.λ0 = 0.0

      try:
         self.ϕ0 = parser.getfloat('Grid', 'ϕ0')
      except (NoOptionError,NoSectionError):
         self.ϕ0 = 0.0

      try:
         self.α0 = parser.getfloat('Grid', 'α0')
      except (NoOptionError,NoSectionError):
         self.α0 = 0.0

      self.nbsolpts         = parser.getint('Spatial_discretization', 'nbsolpts')
      self.nb_elements      = parser.getint('Spatial_discretization', 'nb_elements')
      
      try:
         self.filter_apply     = parser.getint('Spatial_discretization', 'filter_apply') == 1
      except (NoOptionError):
         self.filter_apply     = False
      try:
         self.filter_order     = parser.getint('Spatial_discretization', 'filter_order')
      except (NoOptionError):
         if self.filter_apply:
            self.filter_order  = 16
         else:
            self.filter_order  = 0
      try:
         self.filter_cutoff    = parser.getfloat('Spatial_discretization', 'filter_cutoff')
      except (NoOptionError):
         self.filter_cutoff    = 0

      try:
         self.nb_levels = parser.getint('Spatial_discretization', 'nb_levels')
      except (NoOptionError):
         self.nb_levels = 1

      self.stat_freq   = parser.getint('Output_options', 'stat_freq')
      self.output_freq = parser.getint('Output_options', 'output_freq')
      self.output_file = parser.get('Output_options', 'output_file')
