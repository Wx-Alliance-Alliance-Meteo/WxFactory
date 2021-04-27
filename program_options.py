from configparser import ConfigParser, NoSectionError, NoOptionError

class Configuration:
   def __init__(self, cfg_file: str):

      parser = ConfigParser()
      parser.read(cfg_file)

      print('\nLoading config: ' + cfg_file)
      print(parser._sections)
      print(' ')

      try:
         self.equations = parser.get('General', 'equations')
      except (NoOptionError,NoSectionError):
         self.equations = "shallow water" # TODO : changer

      self.case_number = parser.getint('Test_case', 'case_number')

      if self.case_number == 9:
         self.matsuno_wave_type = parser.get('Test_case', 'matsuno_wave_type')
         self.matsuno_amp = parser.getfloat('Test_case', 'matsuno_amp')

      self.dt               = parser.getfloat('Time_integration', 'dt')
      self.t_end            = parser.getint('Time_integration', 't_end')
      self.time_integrator  = parser.get('Time_integration', 'time_integrator')
      self.krylov_size      = parser.getint('Time_integration', 'krylov_size')
      self.tolerance        = parser.getfloat('Time_integration', 'tolerance')

      try:
         self.ark_solver_exp = parser.get('Time_integration', 'ark_solver_exp')
      except (NoOptionError,NoSectionError):
         self.ark_solver_exp = 'ARK3(2)4L[2]SA-ERK'

      try:
         self.ark_solver_imp = parser.get('Time_integration', 'ark_solver_imp')
      except (NoOptionError,NoSectionError):
         self.ark_solver_imp = 'ARK3(2)4L[2]SA-ESDIRK'

      try:
         self.use_preconditioner = parser.getint('Time_integration', 'use_preconditioner')
      except (NoOptionError, NoSectionError):
         self.use_preconditioner = 0

      try:
         self.precond_filter_before = parser.getint('Time_integration', 'precond_filter_before')
      except (NoOptionError, NoSectionError):
         self.precond_filter_before = 0

      try:
         self.precond_filter_during = parser.getint('Time_integration', 'precond_filter_during')
      except (NoOptionError, NoSectionError):
         self.precond_filter_during = 0

      try:
         self.precond_filter_after = parser.getint('Time_integration', 'precond_filter_after')
      except (NoOptionError, NoSectionError):
         self.precond_filter_after = 0

      try:
         self.discretization = parser.get('Grid', 'discretization')
      except (NoOptionError, NoSectionError):
         self.discretization = 'dg'

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

      try:
         self.ztop = parser.getfloat('Grid', 'ztop')
      except (NoOptionError,NoSectionError):
         self.ztop = 0.0

      self.nbsolpts         = parser.getint('Spatial_discretization', 'nbsolpts')
      self.nb_elements_horizontal = parser.getint('Spatial_discretization', 'nb_elements_horizontal')

      try:
         self.nb_elements_vertical   = parser.getint('Spatial_discretization', 'nb_elements_vertical')
      except (NoOptionError):
         self.nb_elements_vertical   = 1

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

      self.stat_freq   = parser.getint('Output_options', 'stat_freq')
      self.output_freq = parser.getint('Output_options', 'output_freq')
      self.output_file = parser.get('Output_options', 'output_file')
