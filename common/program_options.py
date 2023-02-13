from configparser import ConfigParser, NoSectionError, NoOptionError
import json
from typing       import Any, Dict, List, Optional

__all__ = ['Configuration']

class Configuration:
   """All the config options for GEF"""
   base_output_file: str
   case_number: int
   discretization: str
   exp_smoothe_spectral_radii: List[float]
   exp_smoothe_nb_iters: List[int]
   filter_apply: bool
   nbsolpts: int
   output_dir: str
   preconditioner: str
   sections: Dict

   def __init__(self, cfg_file: str, verbose: bool=True):

      self.sections = {}
      self.parser = ConfigParser()
      self.parser.read(cfg_file)

      if verbose:
         print('\nLoading config: ' + cfg_file)
         print(self.parser._sections)
         print(' ')

      self._get_option('General', 'equations', str, 'euler')

      ################################
      # Test case
      self._get_option('Test_case', 'case_number', int, -1)

      if self.case_number == 9:
         self._get_option('Test_case', 'matsuno_wave_type', str, None)
         self._get_option('Test_case', 'matsuno_amp', float, None)

      self._get_option('Test_case', 'bubble_theta', float, 0.0)
      self._get_option('Test_case', 'bubble_rad', float, 0.0)

      ################################
      # Time integration
      self._get_option('Time_integration', 'dt', float, None)
      self._get_option('Time_integration', 't_end', float, None)
      self._get_option('Time_integration', 'time_integrator', str, None)
      self._get_option('Time_integration', 'tolerance', float, None)

      self._get_option('Time_integration', 'starting_step', int, 0)

      self._get_option('Time_integration', 'exponential_solver', str, 'kiops')
      self._get_option('Time_integration', 'krylov_size', int, 1)
      self._get_option('Time_integration', 'jacobian_method', str, 'complex')

      self._get_option('Time_integration', 'ark_solver_exp', str, 'ARK3(2)4L[2]SA-ERK')
      self._get_option('Time_integration', 'ark_solver_imp', str, 'ARK3(2)4L[2]SA-ESDIRK')

      self._get_option('Time_integration', 'verbose_solver', int, 0)

      ################################
      # Spatial discretization
      self._get_option('Spatial_discretization', 'nbsolpts', int, None)
      self._get_option('Spatial_discretization', 'nb_elements_horizontal', int, None)
      self._get_option('Spatial_discretization', 'nb_elements_vertical', int, 1)
      self.initial_nbsolpts = self.nbsolpts

      self._get_option('Spatial_discretization', 'filter_apply', bool, False)
      self._get_option('Spatial_discretization', 'filter_order', int, 16 if self.filter_apply else 0)
      self._get_option('Spatial_discretization', 'filter_cutoff', float, 0.0)

      ###############################
      # Grid
      possible_grid_types = ['cubed_sphere', 'cartesian2d']
      self._get_option('Grid', 'grid_type', str, 'cubed_sphere', valid_values=possible_grid_types)
      self._get_option('Grid', 'discretization', str, 'dg', ['dg', 'fv'])

      if self.discretization == 'fv':
         if self.nbsolpts != 1:
            raise ValueError(f'The number of solution of solution points ({self.nbsolpts}) in configuration file'
                              ' is inconsistent with a finite volume discretization')

      # Cubed sphere grid params
      self._get_option('Grid', 'λ0', float, 0.0)
      self._get_option('Grid', 'ϕ0', float, 0.0)
      self._get_option('Grid', 'α0', float, 0.0)
      self._get_option('Grid', 'ztop', float, 0.0)

      # Cartesian grid bounds
      self._get_option('Grid', 'x0', float, 0.0)
      self._get_option('Grid', 'x1', float, 0.0)
      self._get_option('Grid', 'z0', float, 0.0)
      self._get_option('Grid', 'z1', float, 0.0)

      ###################
      # Preconditioning
      available_preconditioners = ['none', 'fv', 'fv-mg', 'p-mg']
      self._get_option('Preconditioning', 'preconditioner', str, 'none', valid_values=available_preconditioners)

      self._get_option('Preconditioning', 'num_mg_levels', int, 1, min_value=1)
      if 'mg' not in self.preconditioner: self.num_mg_levels = 1

      self._get_option('Preconditioning', 'precond_tolerance', float, 1e-1)
      self._get_option('Preconditioning', 'num_pre_smoothe', int, 1, min_value=0)
      self._get_option('Preconditioning', 'num_post_smoothe', int, 1, min_value=0)

      self.possible_smoothers = ['exp', 'kiops', 'erk3', 'erk1']
      self._get_option('Preconditioning', 'mg_smoother', str, 'exp', valid_values=self.possible_smoothers)

      self._get_option('Preconditioning', 'exp_smoothe_spectral_radii', List[float], [2.0])
      self.exp_smoothe_spectral_radius = self.exp_smoothe_spectral_radii[0]
      self._get_option('Preconditioning', 'exp_smoothe_nb_iters', List[int], [4])
      self.exp_smoothe_nb_iter = self.exp_smoothe_nb_iters[0]

      self._get_option('Preconditioning', 'mg_solve_coarsest', bool, False)
      self._get_option('Preconditioning', 'kiops_dt_factor', float, 1.1)
      self._get_option('Preconditioning', 'verbose_precond', int, 0)

      ok_interps = ['l2-norm', 'lagrange']
      self._get_option('Preconditioning', 'dg_to_fv_interp', str, 'lagrange', valid_values=ok_interps)
      self._get_option('Preconditioning', 'pseudo_cfl', float, 1.0)

      ###############################
      # Output options
      self._get_option('Output_options', 'stat_freq',       int, 0) # Frequency in timesteps at which to print block stats
      self._get_option('Output_options', 'output_freq',     int, 0) # Frequency in timesteps at which to store the solution
      self._get_option('Output_options', 'save_state_freq', int, 0) # Frequency in timesteps at which to save the state vector
      self._get_option('Output_options', 'store_solver_stats', bool, False) # Whether to store solver stats (at every timestep)

      self._get_option('Output_options', 'output_dir', str, 'results')   # Directory where to store all the output
      self._get_option('Output_options', 'base_output_file', str, 'out') # Base name of file where to store the solution
      self.output_file = f'{self.output_dir}/{self.base_output_file}.nc'

      if verbose: print(f'{self}')

   def _get_option(self, section_name: str, option_name: str, option_type: type, default_value: Any,
                   valid_values: Optional[List]=None, min_value: Any=None, max_value: Any=None):
      value = None
      try:
         if option_type == float:
            value = self.parser.getfloat(section_name, option_name)
         elif option_type == int:
            value = self.parser.getint(section_name, option_name)
         elif option_type == str:
            value = self.parser.get(section_name, option_name).lower()
         elif option_type == bool:
            value = (self.parser.getint(section_name, option_name) > 0)
         elif option_type == List[float]:
            try:
               value = [self.parser.getfloat(section_name, option_name)]
            except ValueError:
               value = [float(x) for x in json.loads(self.parser.get(section_name, option_name))]
         elif option_type == List[int]:
            try:
               value = [self.parser.getint(section_name, option_name)]
            except ValueError:
               value = [int(x) for x in json.loads(self.parser.get(section_name, option_name))]
         else:
            raise ValueError(f'Cannot get this option type (not implemented)')

         # Validate option value
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

      except (NoOptionError, NoSectionError) as e:
         if default_value is None:
            e.message += f"\nMust specify a value for option '{option_name}'"
            raise
         value = default_value

      setattr(self, option_name, value)
      if section_name not in self.sections: self.sections[section_name] = []
      self.sections[section_name].append(option_name)

   def __str__(self):
      out = 'Configuration: \n'
      for section_name, section_options in self.sections.items():
         out += f' - {section_name}'
         long_options = {}
         i = 0
         for option in sorted(section_options):
            val = str(getattr(self, option))
            if len(option) < 28 and len(val) < 12:
               if i % 2 == 0: out += '\n'
               out += f' | {option:27s}: {val:11s}'
               i += 1
            else:
               long_options[option] = val
               # i = 1

         for name, val in long_options.items():
            out += f'\n | {name:27s}: {val}'
         out += '\n'

      return out
