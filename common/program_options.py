from configparser import ConfigParser, NoSectionError, NoOptionError
import json
from typing       import Any, Dict, List, Optional, Type, TypeVar, Union

__all__ = ['Configuration']

OptionType = TypeVar('OptionType', bound=Union[int, float, str, bool, List[int], List[float]])

class Configuration:
   """All the config options for GEF"""
   sections: Dict

   def __init__(self, cfg_file: str, verbose: bool=True):

      self.sections = {}
      self.parser = ConfigParser()
      self.parser.read(cfg_file)

      if verbose:
         print('\nLoading config: ' + cfg_file)
         print(self.parser.sections())
         print(' ')

      self.equations = self._get_option('General', 'equations', str, None, ['euler', 'shallow_water'])
      if self.equations == 'euler':
         self.depth_approx = self._get_option('General', 'depth_approx', str, 'deep', ['deep','shallow'])
      else:
         self.depth_approx = None

      ################################
      # Test case
      self.case_number = self._get_option('Test_case', 'case_number', int, -1)

      if self.case_number == 9:
         self.matsuno_wave_type = self._get_option('Test_case', 'matsuno_wave_type', str, None)
         self.matsuno_amp       = self._get_option('Test_case', 'matsuno_amp', float, None)

      self.bubble_theta = self._get_option('Test_case', 'bubble_theta', float, 0.0)
      self.bubble_rad   = self._get_option('Test_case', 'bubble_rad', float, 0.0)

      ################################
      # Time integration
      self.dt              = self._get_option('Time_integration', 'dt', float, None)
      self.t_end           = self._get_option('Time_integration', 't_end', float, None)
      self.time_integrator = self._get_option('Time_integration', 'time_integrator', str, None)
      self.tolerance       = self._get_option('Time_integration', 'tolerance', float, None)

      self.starting_step = self._get_option('Time_integration', 'starting_step', int, 0)

      self.exponential_solver = self._get_option('Time_integration', 'exponential_solver', str, 'kiops')
      self.krylov_size        = self._get_option('Time_integration', 'krylov_size', int, 1)
      self.jacobian_method    = self._get_option('Time_integration', 'jacobian_method', str, 'complex', ['complex', 'fd'])

      self.verbose_solver = self._get_option('Time_integration', 'verbose_solver', int, 0)
      self.gmres_restart  = self._get_option('Time_integration', 'gmres_restart', int, 20)

      ################################
      # Spatial discretization
      self.nbsolpts               = self._get_option('Spatial_discretization', 'nbsolpts', int, None)
      self.nb_elements_horizontal = self._get_option('Spatial_discretization', 'nb_elements_horizontal', int, None)
      self.nb_elements_vertical   = self._get_option('Spatial_discretization', 'nb_elements_vertical', int, 1)
      self.initial_nbsolpts       = self.nbsolpts
      self.nb_elements_horizontal_total = self.nb_elements_horizontal

      self.relief_layer_height      = self._get_option('Spatial_discretization', 'relief_layer_height', int, 0, min_value=0)
      self.nb_elements_relief_layer = self._get_option('Spatial_discretization', 'nb_elements_relief_layer', int, 0, min_value=0)

      self.filter_apply  = self._get_option('Spatial_discretization', 'filter_apply', bool, False)
      self.filter_order  = self._get_option('Spatial_discretization', 'filter_order', int,
                                             default_value = 16 if self.filter_apply else 0)
      self.filter_cutoff = self._get_option('Spatial_discretization', 'filter_cutoff', float,
                                             default_value = 0.25 if self.filter_apply else 0.0)

      self.expfilter_apply = self._get_option('Spatial_discretization', 'expfilter_apply', bool, False)
      self.expfilter_order = self._get_option('Spatial_discretization', 'expfilter_order', int, None if self.expfilter_apply else 0)
      self.expfilter_strength = self._get_option('Spatial_discretization', 'expfilter_strength', float, None if self.expfilter_apply else 0)
      self.expfilter_cutoff = self._get_option('Spatial_discretization', 'expfilter_cutoff', float, None if self.expfilter_apply else 0)
      
      self.apply_sponge = self._get_option('Spatial_discretization', 'apply_sponge', bool, False)
      self.sponge_tscale = self._get_option('Spatial_discretization', 'sponge_tscale', float, 1.0)
      self.sponge_zscale = self._get_option('Spatial_discretization', 'sponge_zscale', float, 0.0)
      
      ###############################
      # Grid
      possible_grid_types = ['cubed_sphere', 'cartesian2d']
      self.grid_type      = self._get_option('Grid', 'grid_type', str, None, valid_values=possible_grid_types)
      self.discretization = self._get_option('Grid', 'discretization', str, 'dg', ['dg', 'fv'])

      if self.discretization == 'fv':
         if self.nbsolpts != 1:
            raise ValueError(f'The number of solution of solution points ({self.nbsolpts}) in configuration file'
                              ' is inconsistent with a finite volume discretization')

      # Cubed sphere grid params
      if self.grid_type == 'cubed_sphere':
         self.λ0   = self._get_option('Grid', 'λ0', float, None)
         self.ϕ0   = self._get_option('Grid', 'ϕ0', float, None)
         self.α0   = self._get_option('Grid', 'α0', float, None)
         self.ztop = self._get_option('Grid', 'ztop', float, 0.0)

      # Cartesian grid bounds
      if self.grid_type == 'cartesian2d':
         self.x0 = self._get_option('Grid', 'x0', float, None)
         self.x1 = self._get_option('Grid', 'x1', float, None)
         self.z0 = self._get_option('Grid', 'z0', float, None)
         self.z1 = self._get_option('Grid', 'z1', float, None)

      ###################
      # Preconditioning
      available_preconditioners = ['none', 'fv', 'fv-mg', 'p-mg', 'lu', 'ilu']
      self.preconditioner = self._get_option('Preconditioning', 'preconditioner', str, 'none', valid_values=available_preconditioners)

      available_fluxes = ['ausm', 'upwind', 'rusanov']
      self.precond_flux = self._get_option('Preconditioning', 'precond_flux', str, available_fluxes[0], valid_values=available_fluxes)

      self.num_mg_levels = self._get_option('Preconditioning', 'num_mg_levels', int, 1, min_value=1)
      if 'mg' not in self.preconditioner: self.num_mg_levels = 1

      self.precond_tolerance = self._get_option('Preconditioning', 'precond_tolerance', float, 1e-1)
      self.num_pre_smoothe   = self._get_option('Preconditioning', 'num_pre_smoothe', int, 1, min_value=0)
      self.num_post_smoothe  = self._get_option('Preconditioning', 'num_post_smoothe', int, 1, min_value=0)

      self.possible_smoothers = ['exp', 'kiops', 'erk3', 'erk1', 'ark3']
      self.mg_smoother = self._get_option('Preconditioning', 'mg_smoother', str, 'exp', valid_values=self.possible_smoothers)

      self.exp_smoothe_spectral_radii  = self._get_option('Preconditioning', 'exp_smoothe_spectral_radii', List[float], [2.0])
      self.exp_smoothe_spectral_radius = self.exp_smoothe_spectral_radii[0]
      self.exp_smoothe_nb_iters        = self._get_option('Preconditioning', 'exp_smoothe_nb_iters', List[int], [4])
      self.exp_smoothe_nb_iter         = self.exp_smoothe_nb_iters[0]

      self.mg_solve_coarsest = self._get_option('Preconditioning', 'mg_solve_coarsest', bool, False)
      self.kiops_dt_factor   = self._get_option('Preconditioning', 'kiops_dt_factor', float, 1.1)
      self.verbose_precond   = self._get_option('Preconditioning', 'verbose_precond', int, 0)

      ok_interps = ['l2-norm', 'lagrange']
      self.dg_to_fv_interp = self._get_option('Preconditioning', 'dg_to_fv_interp', str, 'lagrange', valid_values=ok_interps)
      self.pseudo_cfl      = self._get_option('Preconditioning', 'pseudo_cfl', float, 1.0)

      ###############################
      # Output options

      # Frequency in timesteps at which to print block stats
      self.stat_freq = self._get_option('Output_options', 'stat_freq', int, 0)
      # Frequency in timesteps at which to store the solution
      self.output_freq = self._get_option('Output_options', 'output_freq', int, 0)
      # Frequency in timesteps at which to save the state vector
      self.save_state_freq = self._get_option('Output_options', 'save_state_freq', int, 0)
      # Whether to store solver stats (at every timestep)
      self.store_solver_stats = self._get_option('Output_options', 'store_solver_stats', bool, False)

      # Directory where to store all the output
      self.output_dir = self._get_option('Output_options', 'output_dir', str, 'results')
      # Name of file where to store the solution
      self.base_output_file = self._get_option('Output_options', 'base_output_file', str, 'out')
      self.output_file = f'{self.output_dir}/{self.base_output_file}.nc'

      self.solver_stats_file = self._get_option('Output_options', 'solver_stats_file', str, 'solver_stats.db')

      if verbose: print(f'{self}')

   def _get_opt_from_parser(self, section_name: str, option_name: str, option_type: Type[OptionType]) -> OptionType:
      value: Optional[OptionType] = None
      if option_type == float:
         value = self.parser.getfloat(section_name, option_name)
      elif option_type == int:
         value = self.parser.getint(section_name, option_name)
      elif option_type == str:
         value = self.parser.get(section_name, option_name).lower()
      elif option_type == bool:
         value = (self.parser.getint(section_name, option_name) > 0)
      elif option_type == List[int]:
         try:
            value = [self.parser.getint(section_name, option_name)]
         except ValueError:
            value = [int(x) for x in json.loads(self.parser.get(section_name, option_name))]
      elif option_type == List[float]:
         try:
            value = [self.parser.getfloat(section_name, option_name)]
         except ValueError:
            value = [float(x) for x in json.loads(self.parser.get(section_name, option_name))]
      else:
         raise ValueError(f'Cannot get this option type (not implemented): {option_type}')

      assert (value is not None)
      return value

   def _validate_option(self,
                        option_name: str,
                        value: OptionType,
                        valid_values: Optional[List[OptionType]],
                        min_value: Optional[OptionType],
                        max_value: Optional[OptionType]) -> OptionType:

      if valid_values is not None and value not in valid_values:
         raise ValueError(
            f'"{value}" is not considered a valid value for option "{option_name}".'
            f' Available values are {valid_values}')
      if min_value is not None:
         if value < min_value:
            print(f'WARNING: Adjusting "{option_name}" to min value "{min_value}"')
            value = min_value
      if max_value is not None:
         if value > max_value:
            print(f'WARNING: Adjusting "{option_name}" to max value "{max_value}"')
            value = max_value

      return value

   def _get_option(self,
                   section_name: str,
                   option_name: str,
                   option_type: Type[OptionType],
                   default_value: Optional[OptionType],
                   valid_values: Optional[List[OptionType]]=None,
                   min_value: Optional[OptionType]=None,
                   max_value: Optional[OptionType]=None) -> OptionType:
      value: Optional[OptionType] = None

      try:
         value = self._get_opt_from_parser(section_name, option_name, option_type)
         value = self._validate_option(option_name, value, valid_values, min_value, max_value)
      except (NoOptionError, NoSectionError) as e:
         if default_value is None:
            e.message += f"\nMust specify a value for option '{option_name}'"
            raise
         value = default_value

      setattr(self, option_name, value)
      if section_name not in self.sections: self.sections[section_name] = []
      self.sections[section_name].append(option_name)

      return value

   def __str__(self):
      out = 'Configuration: \n'
      for section_name, section_options in self.sections.items():
         out += '\n'
         out += f'  {" " + section_name + " ":-^80s}  '
         long_options = {}
         i = 0
         for option in section_options:
            val = str(getattr(self, option))
            if len(option) < 28 and len(val) < 12:
               if i % 2 == 0: out += '\n'
               out += f' | {option:27s}: {val:11s}'
               i += 1
            else:
               long_options[option] = val
               # i = 1
         if i % 2 == 1: out += ' |'

         for name, val in long_options.items():
            out += f'\n | {name:27s}: {val}'
         out += '\n'

      return out
