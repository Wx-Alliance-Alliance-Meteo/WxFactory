from configparser import ConfigParser, NoSectionError, NoOptionError
import json

class Configuration:
   def __init__(self, cfg_file: str, noisy: bool=True):

      parser = ConfigParser()
      parser.read(cfg_file)

      if noisy:
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
      self.time_integrator  = parser.get('Time_integration', 'time_integrator').lower()
      self.tolerance        = parser.getfloat('Time_integration', 'tolerance')

      try:
         self.starting_step = parser.getint('Time_integration', 'starting_step')
      except (NoOptionError,NoSectionError):
         self.starting_step = 0
      self.starting_step = max(self.starting_step, 0)

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

      ################################
      # Spatial discretization
      self.nbsolpts               = parser.getint('Spatial_discretization', 'nbsolpts')
      self.nb_elements_horizontal = parser.getint('Spatial_discretization', 'nb_elements_horizontal')
      self.initial_nbsolpts       = self.nbsolpts

      try:
         self.nb_elements_vertical   = parser.getint('Spatial_discretization', 'nb_elements_vertical')
      except (NoOptionError, NoSectionError):
         self.nb_elements_vertical   = 1

      try:
         self.filter_apply     = parser.getint('Spatial_discretization', 'filter_apply') == 1
      except (NoOptionError, NoSectionError):
         self.filter_apply     = False
      try:
         self.filter_order     = parser.getint('Spatial_discretization', 'filter_order')
      except (NoOptionError, NoSectionError):
         if self.filter_apply:
            self.filter_order  = 16
         else:
            self.filter_order  = 0
      try:
         self.filter_cutoff    = parser.getfloat('Spatial_discretization', 'filter_cutoff')
      except (NoOptionError, NoSectionError):
         self.filter_cutoff    = 0

      ###############################
      # Grid
      try:
         self.discretization = parser.get('Grid', 'discretization')
      except (NoOptionError, NoSectionError):
         self.discretization = 'dg'

      if self.discretization == 'fv':
         if self.nbsolpts != 1:
            raise ValueError(f'The number of solution of solution points ({self.nbsolpts}) in configuration file'
                              ' is inconsistent with a finite volume discretization')

      possible_grid_types = ['cubed_sphere', 'cartesian2d']
      try:
         self.grid_type = parser.get('Grid', 'type').lower()
         if self.grid_type not in possible_grid_types:
            print(f'Selected grid type "{self.grid_type}" is not valid. Possible values are {possible_grid_types}')
            raise ValueError
      except (NoOptionError, NoSectionError):
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

      ###################
      # Preconditioning
      available_preconditioners = ['none', 'fv', 'fv-mg', 'p-mg']
      try:
         self.preconditioner = parser.get('Preconditioning', 'preconditioner').lower()
         if self.preconditioner not in available_preconditioners:
            print(f'Warning: chosen preconditioner {self.preconditioner} is not within available preconditioners.'
                  f' Possible values are {available_preconditioners}')
            self.preconditioner = available_preconditioners[0]
      except (NoOptionError, NoSectionError):
         self.preconditioner = available_preconditioners[0]

      try:
         self.num_mg_levels = max(parser.getint('Preconditioning', 'num_mg_levels'), 1)
      except (NoOptionError, NoSectionError):
         self.num_mg_levels = 1

      if 'mg' not in self.preconditioner: self.num_mg_levels = 1

      try:
         self.precond_tolerance = parser.getfloat('Preconditioning', 'precond_tolerance')
      except (NoOptionError, NoSectionError):
         self.precond_tolerance = 1e-1

      try:
         self.num_pre_smoothe = parser.getint('Preconditioning', 'num_pre_smoothe')
      except (NoOptionError, NoSectionError):
         self.num_pre_smoothe = 1

      try:
         self.num_post_smoothe = parser.getint('Preconditioning', 'num_post_smoothe')
      except (NoOptionError, NoSectionError):
         self.num_post_smoothe = 1

      self.possible_smoothers = ['exp', 'kiops', 'erk3', 'erk1']
      try:
         self.mg_smoother = parser.get('Preconditioning', 'mg_smoother')
         if self.mg_smoother not in self.possible_smoothers:
            print(f'Warning: chosen multigrid smoother {self.mg_smoother} is not within available smoothers.'
                  f' Possible values are {self.possible_smoothers}')
            self.mg_smoother = self.possible_smoothers[0]
      except (NoOptionError, NoSectionError):
         self.mg_smoother = self.possible_smoothers[0]

      try:
         self.exp_smoothe_spectral_radii = [parser.getfloat('Preconditioning', 'exp_smoothe_spectral_radii')]
      except ValueError:
         self.exp_smoothe_spectral_radii = [float(x) for x in \
            json.loads(parser.get('Preconditioning', 'exp_smoothe_spectral_radii'))]
      except (NoOptionError, NoSectionError):
         self.exp_smoothe_spectral_radii = [2.0]
      self.exp_smoothe_spectral_radius = self.exp_smoothe_spectral_radii[0]

      try:
         self.exp_smoothe_nb_iters = [parser.getint('Preconditioning', 'exp_smoothe_nb_iters')]
      except ValueError:
         self.exp_smoothe_nb_iters = [int(x) for x in json.loads(parser.get('Preconditioning', 'exp_smoothe_nb_iters'))]
      except (NoOptionError, NoSectionError):
         self.exp_smoothe_nb_iters = [4]
      self.exp_smoothe_nb_iter = self.exp_smoothe_nb_iters[0]

      try:
         self.mg_solve_coarsest = parser.getint('Preconditioning', 'mg_solve_coarsest') >= 1
      except (NoOptionError, NoSectionError):
         self.mg_solve_coarsest = False

      try:
         self.kiops_dt_factor = parser.getfloat('Preconditioning', 'kiops_dt_factor')
      except (NoOptionError, NoSectionError):
         self.kiops_dt_factor = 1.1

      try:
         self.precond_filter_before = parser.getint('Preconditioning', 'precond_filter_before')
         print(f'Warning: preconditioner filter option  is not properly implemented. Please do not use.')
      except (NoOptionError, NoSectionError):
         self.precond_filter_before = 0

      try:
         self.precond_filter_during = parser.getint('Preconditioning', 'precond_filter_during')
         print(f'Warning: preconditioner filter option  is not properly implemented. Please do not use.')
      except (NoOptionError, NoSectionError):
         self.precond_filter_during = 0

      try:
         self.precond_filter_after = parser.getint('Preconditioning', 'precond_filter_after')
         print(f'Warning: preconditioner filter option  is not properly implemented. Please do not use.')
      except (NoOptionError, NoSectionError):
         self.precond_filter_after = 0

      try:
         self.verbose_precond = parser.getint('Preconditioning', 'verbose_precond') >= 1
      except (NoOptionError, NoSectionError):
         self.verbose_precond = False

      try:
         ok_interps = ['l2-norm', 'lagrange']
         self.dg_to_fv_interp = parser.get('Preconditioning', 'dg_to_fv_interp')
         if self.dg_to_fv_interp not in ok_interps:
            print(f'ERROR: invalid interpolation method for DG to FV conversion ({self.dg_to_fv_interp}).'
                  f' Should pick one of {ok_interps}. Choosing "lagrange" as default.')
            self.dg_to_fv_interp = 'lagrange'
      except (NoOptionError, NoSectionError):
         self.dg_to_fv_interp = 'lagrange'

      try:
         self.pseudo_cfl = parser.getfloat('Preconditioning', 'pseudo_cfl')
      except (NoOptionError, NoSectionError):
         self.pseudo_cfl = 1.0

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
         store_solver_stats_val = parser.getint('Output_options', 'store_solver_stats')
         self.store_solver_stats = True if store_solver_stats_val > 0 else False
      except (NoOptionError, NoSectionError):
         self.store_solver_stats = False

      try:
         self.output_dir = parser.get('Output_options', 'output_dir')
      except (NoOptionError, NoSectionError):
         self.output_dir = 'results'

      try:
         self.output_file = f'{self.output_dir}/{parser.get("Output_options", "output_file")}.nc'
      except (NoOptionError, NoSectionError):
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
         f'Precond:         {self.preconditioner}\n' \
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