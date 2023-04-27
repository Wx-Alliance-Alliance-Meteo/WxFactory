#!/usr/bin/env python3
"""
Set of functions to create various plots from GEF solver stats stored in a SQLite database
"""

import sqlite3
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np

np.set_printoptions(precision=2)

matplotlib.use('Agg')  # To be able to make plots even without an X server

db_connection: Optional[sqlite3.Connection] = None
db_cursor    : Optional[sqlite3.Cursor]     = None

# Type hinting
DataSet = Dict[str, Any]
SingleResultSet = List[DataSet]

case_names = {
   -1: 'unspecified',
    2: 'gaussian bubble',
   21: 'mountain wave with shear',
   31: 'gravity wave',
}

class ProblemDesc:
   """Basic information about a problem specification (mostly size)"""
   def __init__(self, grid_type, equations, case_number, order, hori, vert, dt=0) -> None:
      self.grid_type = grid_type
      self.equations = equations
      self.case_number = case_number
      self.order = order
      self.hori = hori
      self.vert = vert
      self.dt = dt

   # def __init__(self, params: Tuple) -> None:
   #    self.grid_type = params[0]
   #    self.equations = params[1]
   #    self.case_number = params[2]
   #    self.order = params[3]
   #    self.hori  = params[4]
   #    self.vert  = params[5]
   #    self.dt = 0
   #    if len(params) >= 7:
   #       self.dt = params[6]

   def __str__(self):
      result = f'{self.grid_type}, {self.equations}, problem: {case_names[self.case_number]}, {self.hori}x{self.vert}x({self.order})'
      if self.dt > 0.0: result += f' - dt {self.dt}'
      return result

   def short_name(self) -> str:
      short_grid_types = {
         'cubed_sphere': 'cs',
         'cartesian2d' : 'c2d',
      }
      short_eqs = {
         'euler': 'eu',
         'shallow_water': 'sw',
      }

      gt = short_grid_types[self.grid_type]
      eq = short_eqs[self.equations]
      name = f'{gt}_{eq}'
      if self.case_number > 0: name += f'_{self.case_number}'
      name += f'_{self.hori}x{self.vert}x{self.order}'
      if self.dt > 0.0: name += f'_{int(self.dt)}'

      return name


class StdevFunc:
   """
   Aggregator for SQLite to compute the standard deviation in a column.
   """
   def __init__(self):
      self.M = 0.0
      self.S = 0.0
      self.k = 1

   def step(self, value):
      """Add one value to the standard deviation computation"""
      if value is None:
         return
      tM = self.M
      self.M += (value - tM) / self.k
      self.S += (value - tM) * (value - self.M)
      self.k += 1

   def finalize(self) -> float:
      """Compute the deviation from the accumulated values"""
      if self.k < 3:
         return 0.0
      return np.sqrt(self.S / (self.k-2))


def mg_color(index):
   """Determine a plot color based on the given index. For the multigrid preconditioner."""
   mg_colors = ['blue', 'green', 'purple', 'turquoise', 'yellowgreen', 'orangered', 'silver', 'magenta', 'orange',
                'violet']
   return mg_colors[index % len(mg_colors)]

def ref_color(index):
   """Determine a plot color based on the given index. For the reference preconditioner."""
   ref_colors = ['orange', 'magenta', 'darkblue', 'limegreen']
   return ref_colors[index % len(ref_colors)]

def get_color(data: DataSet, index: int):
   """Determine a plot color for the given dataset, given a certain index."""
   if data['precond'] == 'none': return 'black'
   if data['precond'] == 'fv': return ref_color(index)
   return mg_color(index)

def get_linestyle(data: DataSet, index: int):
   """Determine a plot linestyle based on the given dataset and index."""
   main_linestyles = [None, ':']
   if data['precond'] == 'none': return main_linestyles[index % len(main_linestyles)]
   return None

def get_marker(data: DataSet) -> Union[str, int, None]:
   """Determine a plot marker based on the given dataset."""
   if data['precond'] == 'none': return 's'
   if data['precond'] == 'fv': return '+'
   if data['precond'] == 'fv-mg':
      if data['mg_smoother'] == 'ark3':
         return '2'
      return '.'
   if data['precond'] == 'p-mg': return 'x'
   return None

def has_diff(datasets: List[SingleResultSet],
             params: List[str],
             conditions: Callable[[DataSet], bool]=lambda d: True) \
      -> bool:
   """Determine whether the given datasets have different values for the given parameter(s), subject to
   certain (optional) conditions. The conditions are specified as a function that takes a dataset and
   returns a boolean.
   """
   val = None
   all_empty = True
   for dataset in datasets:
      if len(dataset) > 0: all_empty = False
      for data in dataset:
         if conditions(data):
            current = [data[p] for p in params if p in data]
            if val is None: val = current
            elif any(a != b for a,b in zip(val, current)): return True

   return all_empty


class FullDataSet:
   """All solver data related to a specific problem.
   Provides functions to plot certain characteristics of that data.
   """

   def __init__(self, prob: ProblemDesc, output_suffix: str = ''):
      """
      Extract the content of the open database into 4 sets: "no precond", "fv precond", "p-MG precond", "FV-MG" precond.
      """
      self.prob = prob
      self.output_suffix = output_suffix

      time_condition = ''
      if prob.dt > 0.0: time_condition = f'initial_dt = {prob.dt} AND '
      time_condition += f'solver_tol < 10.0 AND '

      t0 = time()
      self.no_precond = []
      # self.no_precond = self._extract_result_set(
      #    ['time_integrator', 'solver_tol', 'initial_dt', 'precond'],
      #    time_condition + 'precond = "none"')

      # no_precond_time = no_precond[0]['time_avg']
      # print(f'precond avg = {no_precond_time}')

      t1 = time()
      self.fv_ref = []
      # self.fv_ref = self._extract_result_set(
      #    ['time_integrator', 'solver_tol', 'precond_tol', 'initial_dt', 'precond'],
      #    time_condition + f'precond = "fv"',
      #    num_best=3)

      t2 = time()
      self.p_mg = self._extract_result_set(
         ['solver_tol', 'precond_interp', 'num_mg_levels', 'kiops_dt_factor', 'mg_solve_coarsest', 'precond_tol',
         'num_pre_smoothe', 'num_post_smoothe', 'mg_smoother', 'initial_dt', 'precond', 'time_integrator'],
         time_condition + f'precond = "p-mg"',
         num_best=10)

      t3 = time()
      self.fv_mg = []
      self.fv_mg += self._extract_result_set(
         ['solver_tol', 'precond_interp', 'num_mg_levels', 'kiops_dt_factor', 'mg_solve_coarsest', #'precond_tol',
          'num_pre_smoothe', 'num_post_smoothe', 'mg_smoother', 'initial_dt', 'precond', 'pseudo_cfl',
          'time_integrator'],
         time_condition + f'precond = "fv-mg" AND mg_smoother = "erk1"',
         num_best=4)
      self.fv_mg += self._extract_result_set(
         ['solver_tol', 'precond_interp', 'num_mg_levels', 'kiops_dt_factor', 'mg_solve_coarsest', #'precond_tol',
          'num_pre_smoothe', 'num_post_smoothe', 'mg_smoother', 'initial_dt', 'precond', 'pseudo_cfl',
          'time_integrator'],
         time_condition + f'precond = "fv-mg" AND mg_smoother = "erk3"',
         num_best=2)
      self.fv_mg += self._extract_result_set(
         ['solver_tol', 'precond_interp', 'num_mg_levels', 'kiops_dt_factor', 'mg_solve_coarsest', #'precond_tol',
          'num_pre_smoothe', 'num_post_smoothe', 'mg_smoother', 'initial_dt', 'precond', 'pseudo_cfl',
          'time_integrator'],
         time_condition + f'precond = "fv-mg" AND mg_smoother = "ark3" AND pseudo_cfl > 0',
         num_best=8)
      self.fv_mg += self._extract_result_set(
         ['solver_tol', 'precond_interp', 'num_mg_levels', 'kiops_dt_factor', 'mg_solve_coarsest', #'precond_tol',
          'num_pre_smoothe', 'num_post_smoothe', 'mg_smoother', 'initial_dt', 'precond', 'pseudo_cfl',
          'time_integrator', 'exp_radius_0', 'exp_radius_1', 'exp_radius_2'],
         time_condition + f'precond = "fv-mg" AND mg_smoother = "exp" AND exp_radius_0 > 0 AND mg_solve_coarsest = 1',
         num_best=1)

      t4 = time()

      print(f'extracted results in {t4 - t0:.2f}s ({t1 - t0:.3f}, {t2 - t1:.3f}, {t3 - t2:.3f}, {t4 - t3:.3f})')

      self.same_tol             = not has_diff([self.no_precond, self.fv_ref, self.p_mg, self.fv_mg], ['solver_tol'])
      self.same_dt              = not has_diff([self.no_precond, self.fv_ref, self.p_mg, self.fv_mg], ['initial_dt'])
      self.with_ref_precond_tol = has_diff([self.fv_ref], ['precond_tol'])
      self.same_mg_precond      = not has_diff([self.fv_mg, self.p_mg], ['precond'])
      self.same_fv_smoother     = not has_diff([self.fv_mg], ['mg_smoother'])
      self.same_fv_pseudo_cfl   = not has_diff([self.fv_mg], ['pseudo_cfl'],
                                               conditions=lambda d: d['mg_smoother'] in ['erk1', 'erk3', 'ark3'])
      self.same_fv_num_smoothe  = not has_diff([self.fv_mg],
                                               ['num_pre_smoothe', 'num_post_smoothe', 'mg_solve_coarsest'])
      self.same_fv_precond_tol  = not has_diff([self.fv_mg], ['precond_tol'])

      if self.same_tol:
         self.tol = self._find_first('solver_tol')
      if self.same_dt:
         self.dt  = self._find_first('initial_dt')
      if self.same_fv_smoother:
         self.fv_smoother = self.fv_mg[0]['mg_smoother']
      if self.same_fv_precond_tol:
         self.fv_precond_tol = None
         if 'precond_tol' in self.fv_mg[0]: 
            self.fv_precond_tol = self.fv_mg[0]['precond_tol']
      if self.same_mg_precond:
         self.mg_precond = 'fv' if len(self.fv_mg) > 0 else 'p'
      if self.same_fv_num_smoothe:
         self.fv_num_smoothe = f'{self.fv_mg[0]["num_pre_smoothe"]}/'      \
                               f'{self.fv_mg[0]["num_post_smoothe"]}/'     \
                               f'{self.fv_mg[0]["mg_solve_coarsest"]}'
      if self.same_fv_pseudo_cfl:
         self.fv_pseudo_cfl = 0.0
         if 'pseudo_cfl' in self.fv_mg[0]:
            self.fv_pseudo_cfl = self.fv_mg[0]["pseudo_cfl"]

   def _find_first(self, param: str) -> Any:
      if len(self.no_precond) > 0: return self.no_precond[0][param]
      if len(self.fv_ref) > 0:     return self.fv_ref[0][param]
      if len(self.fv_mg) > 0:      return self.fv_mg[0][param]
      if len(self.p_mg) > 0:       return self.p_mg[0][param]
      return None

   def _make_label(self, data: Dict[str, Any]) -> str:
      """Create a label for a certain dataset within this superset."""
      elements = []
      if data['precond'] == 'none': elements.append(data['time_integrator'])
      elif data['precond'] == 'fv': elements.append(data['precond'])
      else:
         if self.same_mg_precond: elements.append('mg')
         else: elements.append(data['precond'])
      if not self.same_tol: elements.append(f'tol {data["solver_tol"]:.0e}')
      if not self.same_dt: elements.append(f'dt {data["initial_dt"]}')
      if (self.with_ref_precond_tol and data['precond'] == 'fv') or \
         (not self.same_fv_precond_tol and data['precond'] == 'fv-mg'):
         elements.append(f'ptol {data["precond_tol"]:.0e}')
      if data['precond'] == 'fv-mg':
         if not self.same_fv_smoother: elements.append(f'{data["mg_smoother"]}')
         if data['mg_smoother'] in ['erk1', 'erk3', 'ark3']:
            if not self.same_fv_pseudo_cfl:
               cfl = data["pseudo_cfl"]
               if cfl >= 10000.0: elements.append(f'pdt {cfl/1000.0:.0f}k')
               elif cfl >= 100.0: elements.append(f'pdt {cfl:5.0f}')
               else: elements.append(f'pdt {cfl:5.2f}')
         elif data ['mg_smoother'] == 'exp':
            elements.append(f'rad {data["exp_radius_0"]}/{data["exp_radius_1"]}/{data["exp_radius_2"]}')
         if not self.same_fv_num_smoothe:
            elements.append(f'sm {data["num_pre_smoothe"]}/{data["num_post_smoothe"]}/{data["mg_solve_coarsest"]}')
      return ', '.join(elements)

   def _make_title(self, base_title: str) -> str:
      title = f'{base_title}, {self.prob.grid_type}, {self.prob.equations}, {case_names[self.prob.case_number]}\n'
      if self.prob.hori > 0:
         title += f'{self.prob.hori}'
      else:
         title += '##'
      title += f'x{self.prob.vert}x({self.prob.order}) elem'
      if self.same_dt: title += f', dt = {self.dt}'
      if self.same_tol: title += f', tol = {self.tol:.0e}'
      title += '\n'
      if self.same_mg_precond:
         title += f'MG precond: {self.mg_precond}'
         if self.same_fv_smoother: title += f', smoother {self.fv_smoother}'
         if self.same_fv_num_smoothe: title += f', #sm {self.fv_num_smoothe}'
         if self.same_fv_precond_tol and self.fv_precond_tol is not None: title += f', tol {self.fv_precond_tol}'
         if self.same_fv_pseudo_cfl:
            cfl = self.fv_pseudo_cfl
            if cfl >= 10000.0: title += f', pdt {cfl/1000.0:.0f}k'
            elif cfl >= 100.0: title += f', pdt {cfl:5.0f}'
            else: title += f', pdt {cfl:5.2f}'

      return title

   def _make_filename(self, base_name: str):
      base_filename = f'{base_name}_{self.prob.short_name()}'
      if self.output_suffix != '':
         base_filename += f'_{self.output_suffix}'
      return f'{base_filename}.pdf'

   def _extract_result_set(self,
                           columns: List[str],
                           custom_condition: str,
                           debug: bool = False,
                           num_best: int = 20) \
                            -> SingleResultSet:
      """
      Extract results for configurations corresponding to the parameters stored in [self] and additional optional
      conditions.

      Arguments:
      columns          -- Which additional columns you want to have in the set of results
                          We only select problem configurations based on these columns. If they have the same
                          values for allll of these columns, they will be bunched together as one, even if
                          they vary in other, unselected columns. This is why it can be important to have...
                          custom conditions!
      custom_condition -- Only select results from problem configurations that include this condition
      debug            -- Whether to print debug information
      num_best         -- How many results to include in the set. We take the [num_best] results with
                          the lowest average solve time (averaged in each configuration over all its
                          timesteps)
      """

      # Query to select a subtable containing all columns, for only the problems that have the specified
      # characteristics (size, equations, grid type, etc) and the specified custom condition
      base_subtable_query = f'''
         select *
         from results_param
         where
            grid_type like '{self.prob.grid_type}' AND
            equations like '{self.prob.equations}' AND
            case_number = {self.prob.case_number}  AND
            dg_order    = {self.prob.order}        AND
         {f'num_elem_h  = {self.prob.hori}         AND' if self.prob.hori > 0 else ''}
            num_elem_v  = {self.prob.vert}         AND
            solver_flag = 0                        AND
            {{custom_condition}}
      '''

      # Query to average out some values when a simulation has been run multiple times with the exact same
      # parameters. The average is done timestep by timestep (so we take the average of all first timesteps
      # over the various simulations, then average of all second timesteps, etc.)
      # This is so that we only have 1 value per timestep for each problem configuration
      # This had better give a constant value for the number of iterations...
      combine_same_configs_query = f'''
         select *,
                avg(total_solve_time) as step_time,
                avg(num_solver_it)    as step_it,
                avg(dt)               as step_dt
         from ({base_subtable_query})
         group by {{columns}}, simulation_time, num_elem_h
         order by num_elem_h, num_mg_levels, mg_solve_coarsest, kiops_dt_factor, (num_pre_smoothe + num_post_smoothe),
                  step_id
      '''.strip().format(columns = ', '.join(columns), custom_condition = custom_condition)

      # Query to compute statistics from multiple time steps for each problem configuration that
      # was found in the subtable (according to the given columns)
      param_query = f'''
         select 
                {{columns}},
                num_elem_h,
                avg(step_time)     as avg_step_time,
                stdev(step_time)   as stdev_step_time,
                avg(step_it)       as avg_step_it,
                stdev(step_it)     as stdev_step_time,
                group_concat(simulation_time),
                group_concat(step_dt),
                group_concat(step_time),
                group_concat(step_it)
         from ({combine_same_configs_query})
         group by {{columns}}, num_elem_h
         order by num_mg_levels, mg_solve_coarsest, kiops_dt_factor, (num_pre_smoothe + num_post_smoothe)
      '''.strip().format(columns=', '.join(columns), custom_condition=custom_condition)

      grid_size_tendencies_query = f'''
         select {{columns}},
                min(avg_step_time),
                group_concat(num_elem_h),
                group_concat(avg_step_time),
                group_concat(stdev_step_time),
                group_concat(avg_step_it),
                group_concat(stdev_step_time)
         from ({param_query})
         group by {{columns}}
         order by num_elem_h
      '''.format(columns=', '.join(columns))

      # Query to retrieve and compute the average residual evolution for the selected problem configurations
      # This computes the average residual after each iteration, averaged over every timestep, for a certain
      # configuration
      base_residual_query = f'''
         select iteration, avg(residual), stdev(residual)
         from results_data
         where run_id in (
            with subtable as ({base_subtable_query})
            select distinct run_id
            from subtable
            where {{inner_condition}}
         )
         group by iteration
         order by run_id, iteration
      '''.strip()

      # Retrieve the desired problem configurations and their stats
      if self.prob.hori > 0:
         param_sets = db_cursor.execute(param_query).fetchall()
      else:
         param_sets = db_cursor.execute(grid_size_tendencies_query).fetchall()

      # Print debug info
      if debug:
         subtable_query = base_subtable_query.format(custom_condition = custom_condition)
         subtable_result = db_cursor.execute(subtable_query).fetchall()
         print(f'subtable query: \n{subtable_query}')
         print('\n'.join([f'{r}' for r in subtable_result]))

         combine_result = db_cursor.execute(combine_same_configs_query).fetchall()
         print(f'combine same configs query: \n{combine_same_configs_query}')
         print('\n'.join([f'{r}' for r in combine_result]))

         param_result = db_cursor.execute(param_query).fetchall()
         print(f'param query: \n{param_query}')
         print('\n'.join([f'{r}' for r in param_result]))

         grid_size_result = db_cursor.execute(grid_size_tendencies_query).fetchall()
         print(f'grid size query: \n{grid_size_tendencies_query}')
         print('\n'.join([f'{r}' for r in grid_size_result]))

      # Store results in a list, with the data as their proper type
      results = []
      for subset in param_sets:
         r = {}
         for i, c in enumerate(columns):
            r[c] = subset[i]
         if self.prob.hori > 0:
            # Each problem size is on its own plot
            r['time_avg']      = subset[-8]
            r['time_stdev']    = subset[-7]
            r['it_avg']        = subset[-6]
            r['it_stdev']      = subset[-5]
            r['sim_times']     = np.array([float(x) for x in subset[-4].split(',')])
            r['step_dts']      = np.array([float(x) for x in subset[-3].split(',')])
            r['time_per_step'] = np.array([float(x) for x in subset[-2].split(',')])
            r['it_per_step']   = np.array([float(x) for x in subset[-1].split(',')])
         else:
            # Plot evolution as a function of problem size
            r['time_avg'] = subset[-6] # We use min time as a measure of performance for a certain problem config
            r['num_elem_h']       = np.array([int(x) for x in subset[-5].split(',')])
            r['step_times']       = np.array([float(x) for x in subset[-4].split(',')])
            r['step_times_stdev'] = np.array([float(x) for x in subset[-3].split(',')])
            r['step_its']         = np.array([float(x) for x in subset[-2].split(',')])
            r['step_its_stdev']   = np.array([float(x) for x in subset[-1].split(',')])
         results.append(r)

      # Only select the fastest problem configurations (b/c the following step can be expensive)
      if num_best > 0:
         num_best = min(num_best, len(param_sets))
         results = sorted(results, key=lambda d: d['time_avg'])[:num_best]

      # Get the residuals from the other table
      if self.prob.hori > 0:
         for subset in results:
            if debug:
               print(f'subset = \n{subset}')

            inner_cond = ' AND '.join([f'{c} = "{subset[c]}"' for c in columns])
            residual_query = base_residual_query.format(inner_condition=inner_cond, custom_condition=custom_condition)
            residuals = db_cursor.execute(residual_query).fetchall()

            subset['residuals'] = np.array([r[1] for r in residuals])
            subset['stdevs']    = np.array([r[2] if r[2] is not None else 0.0 for r in residuals])

      return results

   def plot_time_per_step(self):
      """Plot solve time for each time step of the simulation (average, when it was run multiple times)."""

      fig, ax = plt.subplots(1, 1)

      for dataset in [self.no_precond, self.fv_ref, self.p_mg, self.fv_mg]:
         for i, data in enumerate(dataset):
            ax.plot(data['sim_times'] + data['step_dts'], data['time_per_step'] / data['step_dts'],
                  color=get_color(data, i), linestyle=get_linestyle(data, i), marker=get_marker(data),
                  label=self._make_label(data))

      # ax.set_ylim(bottom=0)
      ax.yaxis.grid()

      ax.set_xlabel(f'Problem time (s)')
      ax.set_ylabel('Solve time (s/s)')

      fig.suptitle(self._make_title('Solver time'), fontsize=11, x=0.12, horizontalalignment='left')
      fig.legend(fontsize=8)

      full_filename = self._make_filename('time_per_step')
      fig.savefig(full_filename)
      plt.close(fig)
      print(f'Saving {full_filename}')

   def plot_it_per_size(self):
      """Plot the number of iterations per time step with respect to problem size."""
      fig, ax = plt.subplots(1, 1)
      for dataset in [self.no_precond, self.fv_ref, self.p_mg, self.fv_mg]:
         for i, data in enumerate(dataset):
            ax.errorbar(data['num_elem_h'], y=data['step_its'], yerr=data['step_its_stdev'],
                  color=get_color(data, i), linestyle=get_linestyle(data, i), marker=get_marker(data),
                  label=self._make_label(data))

      ax.yaxis.grid()

      ax.set_xlabel(f'Problem size (num elem horizontally)')
      ax.set_ylabel('Number of iterations')

      fig.suptitle(self._make_title('# Iterations'), fontsize=11, x=0.12, horizontalalignment='left')
      fig.legend(fontsize=8)

      full_filename = self._make_filename('it_per_size')
      fig.savefig(full_filename)
      plt.close(fig)
      print(f'Saving {full_filename}')

   def plot_it_per_step(self):
      """Plot solve time for each time step of the simulation (average, when it was run multiple times)."""

      fig, ax = plt.subplots(1, 1)

      for dataset in [self.no_precond, self.fv_ref, self.p_mg, self.fv_mg]:
         for i, data in enumerate(dataset):
            ax.plot(data['sim_times'] + data['step_dts'], data['it_per_step'],
                  color=get_color(data, i), linestyle=get_linestyle(data, i), marker=get_marker(data),
                  label=self._make_label(data))

      ax.yaxis.grid()
      ax.set_xlabel('Time step')
      ax.set_ylabel('Number of iterations')
      # ax.set_yscale('log', base=2)
      ax.tick_params(axis='y', which='minor')

      # # Set bounds on y axis
      # ymin, ymax = ax.get_ylim()
      # ymin = min(ymin, 10)
      # if ymax < 100:
      #    ymax = 100
      # elif ymax < 1000:
      #    ymax = 1000
      # ax.set_ylim(bottom=ymin, top=ymax)

      fig.suptitle(self._make_title('Solver iterations'), fontsize=11, x=0.12, horizontalalignment='left')
      fig.legend(fontsize=8)

      full_filename = self._make_filename('it_per_step')
      fig.savefig(full_filename)
      plt.close(fig)
      print(f'Saving {full_filename}')

   def plot_residual(self):
      """
      Plot the evolution of the residual per FGMRES iteration. Since it varies by time step, we plot the average per
      iteration, with error bars for standard deviation
      """

      MAX_IT  = 50
      max_res = 0
      min_res = np.inf
      max_it  = 0

      fig, ax_res = plt.subplots(1, 1)

      for dataset in [self.no_precond, self.fv_ref, self.p_mg, self.fv_mg]:
         for i, data in enumerate(dataset):
            ax_res.errorbar(np.arange(len(data['residuals'][:MAX_IT])),
               y = data['residuals'][:MAX_IT], yerr = data['stdevs'][:MAX_IT],
                  color=get_color(data, i), linestyle=get_linestyle(data, i), marker=get_marker(data),
                  label=self._make_label(data))

      ax_res.set_xlabel('Iteration #')
      if max_it > 0: ax_res.set_xlim(left = -1, right = min(max_it, MAX_IT))
      # ax_res.set_xscale('log')

      ax_res.set_yscale('log')
      ax_res.grid(True)
      ax_res.set_ylabel('Residual')
      if max_res > 0.0: ax_res.set_ylim(bottom = max(min_res * 0.8, 5e-8), top = min(2e0, max_res * 1.5))

      fig.suptitle(self._make_title('Residual evolution'), fontsize=11, x=0.12, horizontalalignment='left')
      fig.legend(fontsize=8)

      full_filename = self._make_filename('residual')
      fig.savefig(full_filename)
      plt.close(fig)
      print(f'Saving {full_filename}')

   def plot_time(self):
      """
      Plot a bar chart of the solve time for each set of parameters, averaged over multiple time steps, with error bars
      showing standard deviation
      """

      fig, ax_time = plt.subplots(1, 1, figsize=(20, 9))

      names = []
      times = []
      errors = []

      for dataset in [self.no_precond, self.fv_ref, self.fv_mg, self.p_mg]:
         for data in dataset:
            names.append(self._make_label(data))
            times.append(data['time_avg'])
            errors.append(data['time_stdev'])

      ax_time.barh(names, times, xerr=errors, log=False)
      ax_time.invert_yaxis()
      ax_time.xaxis.set_minor_locator(AutoMinorLocator())
      ax_time.xaxis.grid(True, which='both')
      ax_time.xaxis.grid(True, which='minor', linestyle='dotted')
      ax_time.set_xlabel('seconds')

      fig.suptitle(self._make_title('Solver time'))

      full_filename = self._make_filename('avg_time')
      fig.savefig(full_filename)
      plt.close(fig)
      print(f'Saving {full_filename}')

   def plot_it(self):
      """
      Plot a bar chart of the solve time for each set of parameters, averaged over multiple time steps, with error bars
      showing standard deviation
      """
      fig, ax = plt.subplots(1, 1, figsize=(20, 9))

      names = []
      times = []
      errors = []

      for dataset in [self.no_precond, self.fv_ref, self.fv_mg, self.p_mg]:
         for data in dataset:
            names.append(self._make_label(data))
            times.append(data['it_avg'])
            errors.append(data['it_stdev'])

      ax.barh(names, times, xerr=errors, log=False)
      ax.invert_yaxis()
      ax.xaxis.set_minor_locator(AutoMinorLocator())
      ax.xaxis.grid(True, which='both')
      ax.xaxis.grid(True, which='minor', linestyle='dotted')
      ax.set_xlabel('iterations')

      fig.suptitle(self._make_title('Solver iterations'))

      full_filename = self._make_filename('avg_it')
      fig.savefig(full_filename)
      plt.close(fig)
      print(f'Saving {full_filename}')

   def plot_error_time(self):
      """Plot the evolution of the residual over time (rather than iterations)."""

      plot_work = False

      fig: Optional[matplotlib.figure.Figure] = None
      ax_time: Optional[plt.Axes] = None
      ax_work: Optional[plt.Axes] = None

      if plot_work:
         fig, (ax_time, ax_work) = plt.subplots(2, 1)
      else:
         fig, ax_time = plt.subplots(1, 1)

      for dataset in [self.no_precond, self.fv_ref, self.p_mg, self.fv_mg]:
         for i, data in enumerate(dataset):
            ax_time.plot(
               data['times'], data['residuals'],
               color=get_color(data, i), linestyle=get_linestyle(data, i), marker=get_marker(data),
               label=self._make_label(data))
            if plot_work:
               ax_work.plot(
                  data['work'], data['residuals'],
                  color=get_color(data, i), linestyle=get_linestyle(data, i), marker=get_marker(data),
                  label=self._make_label(data))

      ax_time.grid(True)
      ax_time.set_xlabel('Time (s)')
      # ax_time.set_xlim([1.0, max_time])
      ax_time.set_ylabel('Residual')
      ax_time.set_yscale('log')

      if plot_work:
         # ax_work.set_yscale('log')
         ax_work.grid(True)
         ax_work.set_xlim(left=1.0)
         ax_work.set_xlabel('Work (estimate)')
         ax_work.set_ylabel('Residual')
         ax_work.set_yscale('log')

      fig.suptitle(self._make_title(f'Time (work) to reach accuracy'), fontsize=11, x=0.12, horizontalalignment='left')
      fig.legend(fontsize=8)

      full_filename = self._make_filename('error')
      fig.savefig(full_filename)
      plt.close(fig)
      print(f'Saving {full_filename}')


def main(args):
   """
   Extract data and call appropriate plotting functions.
   """

   if args.split_dt:
      sizes_query = '''
         select distinct grid_type, equations, case_number, dg_order, num_elem_h, num_elem_v, initial_dt
         from results_param
      '''
      probs = [ProblemDesc(p[0], p[1], p[2], p[3], p[4], p[5], p[6])
               for p in list(db_cursor.execute(sizes_query).fetchall())]

   else:
      sizes_query = '''
         select distinct grid_type, equations, case_number, dg_order, num_elem_h, num_elem_v
         from results_param
      '''
      probs = [ProblemDesc(p[0], p[1], p[2], p[3], p[4], p[5])
               for p in list(db_cursor.execute(sizes_query).fetchall())]

   for p in probs: print(f'{p}')

   if any([args.residual, args.time, args.it, args.error_time, args.time_per_step, args.it_per_step]):
      for prob in probs:
         prob_data = FullDataSet(prob, output_suffix=args.suffix)
         if args.residual:      prob_data.plot_residual()
         if args.time:          prob_data.plot_time()
         if args.it:            prob_data.plot_it()
         if args.error_time:    prob_data.plot_error_time()
         if args.time_per_step: prob_data.plot_time_per_step()
         if args.it_per_step:   prob_data.plot_it_per_step()

   if args.grid_progression:
      # Only horizontal progression
      prob_query = '''
         select distinct grid_type, equations, case_number, dg_order, num_elem_v
         from results_param
      '''
      probs = [ProblemDesc(p[0], p[1], p[2], p[3], 0, p[4])
               for p in list(db_cursor.execute(prob_query).fetchall())]
      for p in probs: print(f'{p}')
      for prob in probs:
         prob_data = FullDataSet(prob, output_suffix=args.suffix)
         prob_data.plot_it_per_size()
         break



if __name__ == '__main__':
   import argparse

   parser = argparse.ArgumentParser(description='Plot results from automated preconditioner tests')
   parser.add_argument('results_file', type=str, help='DB file that contains test results')
   parser.add_argument('--residual', action='store_true', help='Plot residual evolution')
   parser.add_argument('--time', action='store_true', help='Plot the time needed to reach target residual')
   parser.add_argument('--it', action='store_true',
                       help='Plot the number of iterations needed to reach target residual')
   parser.add_argument('--error-time', action='store_true',
                       help='Plot time needed to reach a certain error (residual) level')
   parser.add_argument('--time-per-step', action='store_true',
                       help='Plot time for solve per time step (real time over simulation time)')
   parser.add_argument('--it-per-step', action='store_true', help='Plot solver iteration count per time step')
   parser.add_argument('--split-dt', action='store_true',
                       help="Whether to split plots per dt (rather than all dt's on the same plot)")
   parser.add_argument('--suffix', type=str, default='', help='Suffix to add to the base output file name')
   parser.add_argument('--grid-progression', action='store_true', help='oijiojoij')

   parsed_args = parser.parse_args()

   db_connection = sqlite3.connect(parsed_args.results_file)
   db_connection.create_aggregate("stdev", 1, StdevFunc)
   db_cursor = db_connection.cursor()

   main(parsed_args)
