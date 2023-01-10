from configparser import ConfigParser, NoSectionError, NoOptionError

class Configuration:
   def __init__(self, cfg_file: str, noisy: bool=True):

      parser = ConfigParser()
      parser.read(cfg_file)

      if (noisy):
         print('\nLoading config: ' + cfg_file)
         print(parser._sections)
         print(' ')

      try:
         self.equations = parser.get('General', 'equations').lower()
      except (NoOptionError,NoSectionError):
         self.equations = "euler"

      ################################
      # Test case
      self.case_number = parser.getint('Test_case', 'case_number')

      if self.case_number == 9:
         self.matsuno_wave_type = parser.get('Test_case', 'matsuno_wave_type')
         self.matsuno_amp = parser.getfloat('Test_case', 'matsuno_amp')

      self.bubble_theta = 0.0
      self.bubble_rad   = 0.0
      try:
         self.bubble_theta    = parser.getfloat('Test_case', 'bubble_theta')
         self.bubble_rad      = parser.getfloat('Test_case', 'bubble_rad')
      except(NoOptionError,NoSectionError):
         pass

      ################################
      # Time integration
      self.dt               = parser.getfloat('Time_integration', 'dt')
      self.t_end            = parser.getint('Time_integration', 't_end')
      self.time_integrator  = parser.get('Time_integration', 'time_integrator')
      self.tolerance        = parser.getfloat('Time_integration', 'tolerance')

      try:
         self.starting_step = parser.getint('Time_integration', 'starting_step')
      except:
         self.starting_step = 0

      try:
         self.exponential_solver = parser.get('Time_integration', 'exponential_solver')
      except (NoOptionError,NoSectionError):
         self.exponential_solver = 'kiops'

      try:
         self.krylov_size = parser.getint('Time_integration', 'krylov_size')
      except (NoOptionError,NoSectionError):
         self.krylov_size = 1

      try:
         self.jacobian_method = parser.get('Time_integration', 'jacobian_method')
      except (NoOptionError,NoSectionError):
         self.jacobian_method = 'complex'

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

      ###############################
      # Grid
      try:
         self.discretization = parser.get('Grid', 'discretization')
      except (NoOptionError, NoSectionError):
         self.discretization = 'dg'

      if self.discretization == 'fv':
         if self.nbsolpts != 1:
            if (noisy):
               print('The number of solution of solution points in configuration file is inconsistent with a finite volume discretization')
            exit(0)

      possible_grid_types = ['cubed_sphere', 'cartesian2d']
      try:
         self.grid_type = parser.get('Grid', 'type').lower()
         if self.grid_type not in possible_grid_types:
            print(f'Selected grid type "{self.grid_type}" is not valid. Possible values are {possible_grid_types}')
            raise ValueError
      except:
         self.grid_type = 'cubed_sphere'

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

      # Cartesian grid bounds
      self.x0 = 0.0
      self.x1 = 0.0
      self.z0 = 0.0
      self.z1 = 0.0
      try: 
         self.x0 = parser.getfloat('Grid', 'x0')
         self.x1 = parser.getfloat('Grid', 'x1')
         self.z0 = parser.getfloat('Grid', 'z0')
         self.z1 = parser.getfloat('Grid', 'z1')
      except(NoOptionError, NoSectionError):
         pass

      ################################
      # Spatial discretization
      self.nbsolpts               = parser.getint('Spatial_discretization', 'nbsolpts')
      self.nb_elements_horizontal = parser.getint('Spatial_discretization', 'nb_elements_horizontal')
      self.initial_nbsolpts       = self.nbsolpts

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

      ###############################
      # Output options

      self.stat_freq   = parser.getint('Output_options', 'stat_freq')
      self.output_freq = parser.getint('Output_options', 'output_freq')

      # Frequency (in timesteps) at which to save the state vector of the simulation
      try:
         self.save_state_freq = parser.getint('Output_options', 'save_state_freq')
      except (NoOptionError, NoSectionError):
         self.save_state_freq = 0

      try:
         self.output_dir = parser.get('Output_options', 'output_dir')
      except:
         self.output_dir = 'results'

      try:
         self.output_file = f'{self.output_dir}/{parser.get("Output_options", "output_file")}.nc'
      except:
         self.output_file = f'{self.output_dir}/out.nc'

   def __str__(self):
      return \
         f'Equations: {self.equations}\n' \
         f'Case number: {self.case_number}\n' \
         f'dt:          {self.dt}\n' \
         f't_end:       {self.t_end}\n' \
         f'Time integrator: {self.time_integrator}\n' \
         f'Krylov size:     {self.krylov_size}\n' \
         f'tolerance:       {self.tolerance}\n' \
         f'ARK solver exp:  {self.ark_solver_exp}\n' \
         f'ARK solver imp:  {self.ark_solver_imp}\n' \
         f'Use precond:     {self.use_preconditioner}\n' \
         f'Precond filter \n  before: {self.precond_filter_before}\n  during: {self.precond_filter_during}\n  after:  {self.precond_filter_after}\n' \
         f'Discretization:  {self.discretization}\n' \
         f'λ0: {self.λ0}\n' \
         f'ϕ0: {self.ϕ0}\n' \
         f'α0: {self.α0}\n' \
         f'ztop: {self.ztop}\n' \
         f'initial nbsolpts: {self.initial_nbsolpts}\n' \
         f'nbsolpts: {self.nbsolpts}\n' \
         f'# elem horiz: {self.nb_elements_horizontal}\n' \
         f'# elem vert:  {self.nb_elements_vertical}\n' \
         f'apply filter: {self.filter_apply}\n' \
         f'filer order:  {self.filter_order}\n' \
         f'filter cutoff: {self.filter_cutoff}\n' \
         f'stat frequency: {self.stat_freq}\n' \
         f'output frequency: {self.output_freq}\n' \
         f'output file:      {self.output_file}\n'
