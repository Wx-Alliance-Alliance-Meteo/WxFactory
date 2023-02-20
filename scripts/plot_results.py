#!/usr/bin/env python3
"""
Set of functions to create various plots from GEF solver stats stored in a SQLite database
"""

import sqlite3
from time import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np

matplotlib.use('Agg')  # To be able to make plots even without an X server

db_connection: Optional[sqlite3.Connection] = None
db_cursor    : Optional[sqlite3.Cursor]     = None

# Type hinting
SingleResultSet = List[Dict[str, Any]]

class ProblemDesc:
   def __init__(self, params: Tuple) -> None:
      self.grid_type = params[0]
      self.equations = params[1]
      self.order = params[2]
      self.hori  = params[3]
      self.vert  = params[4]
      self.dt = 0
      if len(params) >= 6:
         self.dt = params[5]

   def __str__(self):
      result = f'{self.grid_type}, {self.equations}, {self.hori}x{self.vert}x({self.order})'
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
      name = f'{gt}_{eq}_{self.hori}x{self.vert}x{self.order}'
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

def extract_results(prob: ProblemDesc) -> Tuple[SingleResultSet, SingleResultSet, SingleResultSet, SingleResultSet]:
   """
   Extract the content of the open database into 4 sets: "no precond", "fv precond", "p-MG precond", "FV-MG" precond.
   """

   def get_single_precond_results(columns, custom_condition, debug=False) \
         -> SingleResultSet:
      """
      Arguments:
      columns          -- Which additional columns you want to have in the set of results
      custom_condition -- A custom condition to be included in the query
      """

      # Query to select a subtable containing all columns, for only the problems that have the specified
      # size/order/timestep and the specified custom condition
      base_subtable_query = f'''
         select * from results_param
         where
            grid_type like '{prob.grid_type}' AND
            equations like '{prob.equations}' AND
            dg_order   = {prob.order}       AND
            num_elem_h = {prob.hori}        AND
            num_elem_v = {prob.vert}        AND
            {{custom_condition}}
      '''

      time_per_step_query = f'''
         select *, avg(total_solve_time) as step_time, avg(num_solver_it) as step_it, avg(dt) as step_dt
         from ({base_subtable_query})
         group by {{columns}}, simulation_time
         order by num_mg_levels, mg_solve_coarsest, kiops_dt_factor, (num_pre_smoothe + num_post_smoothe), step_id
      '''.strip().format(columns = ', '.join(columns), custom_condition = custom_condition)

      # Query to compute statistics from multiple time steps for each set of precondtioner parameters that
      # was found in the subtable (according to the given columns)
      base_param_query = f'''
         select {{columns}},
                group_concat(simulation_time),
                group_concat(step_dt),
                group_concat(step_time),
                group_concat(step_it),
                avg(step_time),
                stdev(step_time),
                avg(step_it),
                stdev(step_it)
         from ({time_per_step_query})
         group by {{columns}}
         order by num_mg_levels, mg_solve_coarsest, kiops_dt_factor, (num_pre_smoothe + num_post_smoothe)
      '''.strip()

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

      results = []
      param_query = base_param_query.format(columns = ', '.join(columns), custom_condition = custom_condition)
      param_sets = db_cursor.execute(param_query).fetchall()

      if debug:
         subtable_query = base_subtable_query.format(custom_condition = custom_condition)
         subtable_result = db_cursor.execute(subtable_query).fetchall()
         print(f'subtable query: \n{subtable_query}')
         print('\n'.join([f'{r}' for r in subtable_result]))
         print(f'param query: \n{param_query}')

      for subset in param_sets:
         if debug:
            print(f'subset = \n{subset}')
         residual_query = base_residual_query.format(
            inner_condition = ' AND '.join([f'{c} = "{subset[i]}"' for i, c in enumerate(columns)]),
            custom_condition = custom_condition)
         # if debug:
         #    print(f'res query: \n{residual_query}')
         residuals = db_cursor.execute(residual_query).fetchall()

         set_result = {}
         for i, c in enumerate(columns):
            set_result[c] = subset[i]
         set_result['sim_times']     = np.array([float(x) for x in subset[-8].split(',')])
         set_result['step_dts']      = np.array([float(x) for x in subset[-7].split(',')])
         set_result['time_per_step'] = np.array([float(x) for x in subset[-6].split(',')])
         set_result['it_per_step']   = np.array([float(x) for x in subset[-5].split(',')])
         set_result['time_avg']      = subset[-4]
         set_result['time_stdev']    = subset[-3]
         set_result['it_avg']        = subset[-2]
         set_result['it_stdev']      = subset[-1]
         set_result['residuals']     = np.array([r[1] for r in residuals])
         set_result['stdevs']        = np.array([r[2] if r[2] is not None else 0.0 for r in residuals])

         results.append(set_result)

         # print(f'time per step: {set_result["time_per_step"]}')

      return results

   time_condition = ''
   if prob.dt > 0.0: time_condition = f'initial_dt = {prob.dt} AND '

   t0 = time()
   no_precond = get_single_precond_results(
      ['time_integrator', 'solver_tol', 'initial_dt'],
      time_condition + 'precond = "none"')

   # no_precond_time = no_precond[0]['time_avg']
   # print(f'precond avg = {no_precond_time}')

   t1 = time()
   fv_ref = get_single_precond_results(
      ['solver_tol', 'precond_tol'],
      time_condition + f'precond = "fv" AND precond_tol > 1e-2')

   t2 = time()
   p_mg_results = get_single_precond_results(
      ['solver_tol', 'precond_interp', 'num_mg_levels', 'kiops_dt_factor', 'mg_solve_coarsest', 'precond_tol',
       'num_pre_smoothe', 'num_post_smoothe', 'mg_smoother'],
      time_condition + f'precond = "p-mg"'
   )

   t3 = time()
   fv_mg_results = get_single_precond_results(
      ['solver_tol', 'precond_interp', 'num_mg_levels', 'kiops_dt_factor', 'mg_solve_coarsest', 'precond_tol',
       'num_pre_smoothe', 'num_post_smoothe', 'mg_smoother', 'initial_dt'],
      time_condition + f'precond = "fv-mg"'
   )

   t4 = time()

   print(f'extracted results in {t4 - t0:.2f}s ({t1 - t0:.3f}, {t2 - t1:.3f}, {t3 - t2:.3f}, {t4 - t3:.3f})')

   return no_precond, fv_ref, p_mg_results, fv_mg_results


main_linestyles = [None, ':', None, ':']
mg_linestyles = [None, '--', '-.', ':', None, '--', '-.', ':']
# smoothings_linestyles = [':', '-.', '--', None]
smoothings_linestyles = [None, (0, (1, 4)), (0, (1, 2)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (5, 3)),
                         (0, (5, 1)), None]
mg_colors = ['blue', 'green', 'purple', 'teal', 'blue', 'green', 'purple']
ref_colors = ['orange', 'magenta']

def plot_residual(no_precond, fv_ref, p_mg, fv_mg, prob):
   """
   Plot the evolution of the residual per FGMRES iteration. Since it varies by time step, we plot the average per
   iteration, with error bars for standard deviation
   """

   MAX_IT = 30
   max_res = 0
   min_res = np.inf
   max_it = 0

   fig, ax_res = plt.subplots(1, 1)
   for i, data in enumerate(no_precond):
      ax_res.plot(data['residuals'][:MAX_IT], color='black', label=f'No precond, solver {data["linear_solver"]}')
      max_res = max(data['residuals'][:MAX_IT].max(), max_res)
      max_it = len(data['residuals'][:])

   for i, data in enumerate(fv_ref):
      ax_res.errorbar(np.arange(len(data['residuals'][:MAX_IT])),
         y = data['residuals'][:MAX_IT], yerr = data['stdevs'][:MAX_IT],
         color = ref_colors[i%len(ref_colors)], linestyle=mg_linestyles[i%4],
         label=f'Ref precond, tol={data["precond_tol"]:.0e}')
      max_res = max(data['residuals'][:MAX_IT].max(), max_res)
      min_res = min(data['residuals'][:MAX_IT].min(), min_res)

   for i, data in enumerate(p_mg):
      # ax_res.plot(data['residuals'][:MAX_IT], color=mg_colors[data['mg_levels']],
      ax_res.errorbar(x = np.arange(len(data['residuals'][:MAX_IT])),
            y = data['residuals'][:MAX_IT], yerr = data['stdevs'][:MAX_IT],
            color=mg_colors[data['num_mg_levels']], linestyle=mg_linestyles[0],
            label=f'p-MG prec ({data["num_mg_levels"]} lvl, {data["num_pre_smoothe"]}/{data["num_post_smoothe"]} sm)')
      max_res = max(data['residuals'][:MAX_IT].max(), max_res)
      min_res = min(data['residuals'][:MAX_IT].min(), min_res)

   for i, data in enumerate(fv_mg):
      ls = smoothings_linestyles[data['num_pre_smoothe'] + data['num_post_smoothe']]
      # if data['smoother'] == 'irk': ls = ':'
      ax_res.errorbar(
            np.arange(len(data['residuals'][:MAX_IT])),
            data['residuals'][:MAX_IT], yerr = data['stdevs'][:MAX_IT],
            color=mg_colors[data['num_mg_levels']], linestyle=ls,
            label=f'FV-MG prec {data["num_mg_levels"]} lvl, {data["num_pre_smoothe"]}/{data["num_post_smoothe"]} sm')
      max_res = max(data['residuals'][:MAX_IT].max(), max_res)
      min_res = min(data['residuals'][:MAX_IT].min(), min_res)

   ax_res.set_xlabel('Iteration #')
   if max_it > 0: ax_res.set_xlim(left = -1, right = min(max_it, MAX_IT))
   # ax_res.set_xscale('log')

   ax_res.set_yscale('log')
   ax_res.grid(True)
   ax_res.set_ylabel('Residual')
   if max_res > 0.0: ax_res.set_ylim(bottom = max(min_res * 0.8, 5e-8), top = min(2e0, max_res * 1.5))

   fig.suptitle(f'Residual evolution\n{size[0]}x{size[1]}x{size[2]} elem, {size[3]} dt')
   fig.legend()
   full_filename = f'residual_{size[0]}x{size[1]}x{size[2]}_{int(size[3])}.pdf'
   fig.savefig(full_filename)
   plt.close(fig)
   print(f'Saving {full_filename}')

def plot_time(no_precond, fv_ref, p_mg, fv_mg, prob):
   """
   Plot a bar chart of the solve time for each set of parameters, averaged over multiple time steps, with error bars
   showing standard deviation
   """

   factors = [f[0] for f in db_cursor.execute(f'select distinct kiops_dt_factor from results_param').fetchall()]

   for factor in factors:
      fig, ax_time = plt.subplots(1, 1, figsize=(20, 9))

      names = []
      times = []
      errors = []

      for _, data in enumerate(no_precond):
         names.append('no_precond')
         times.append(data['time_avg'])
         errors.append(data['time_stdev'])

      for _, data in enumerate(fv_ref):
         names.append(f'fv, tol={data["precond_tol"]:.0e}')
         times.append(data['time_avg'])
         errors.append(data['time_stdev'])

      for _, data in enumerate(p_mg):
         if data['kiops_dt_factor'] == factor:
            names.append(f'p-MG, {data["num_mg_levels"]} lvl, {data["num_pre_smoothe"]}/{data["num_post_smoothe"]} sm,'
                         f' {data["mg_smoothe_only"]}, {data["precond_tol"]:.0e}')
            times.append(data['time_avg'])
            errors.append(data['time_stdev'])

      for _, _ in enumerate(fv_mg):
         pass

      ax_time.barh(names, times, xerr=errors, log=False)
      ax_time.invert_yaxis()

      fig.suptitle(f'Time to solution for each timestep\n'
                   f'{size[0]}x{size[1]}x{size[2]} elem, {size[3]} dt\n'
                   f'kiops factor {factor}')
      full_filename = f'step_time_{size[0]}x{size[1]}x{size[2]}_{int(size[3])}_{int(factor*100):03d}.pdf'
      fig.savefig(full_filename)
      plt.close(fig)
      print(f'Saving {full_filename}')

def plot_error_time(no_precond, fv_ref, p_mg, fv_mg, prob):
   """
   Plot the evolution of the residual over time (rather than iterations)
   """

   max_time  = 0.0
   max_work  = 0.0
   plot_work = False

   fig: Optional[matplotlib.figure.Figure] = None
   ax_time: Optional[plt.Axes] = None
   ax_work: Optional[plt.Axes] = None

   if plot_work:
      fig, (ax_time, ax_work) = plt.subplots(2, 1)
   else:
      fig, ax_time = plt.subplots(1, 1)

   if len(no_precond) > 0:
      data = no_precond[0]
      ax_time.plot(data['times'], data['residuals'], color='black', label='GMRES')
      max_time = np.max([max_time, data['times'].max()])
      if plot_work:
         ax_work.plot(data['work'], data['residuals'], color='black')
         max_work = np.max([max_work, data['work'].max()])

   max_time *= 2.0
   max_work *= 1.2

   if max_time <= 0.0: max_time = 2000.0

   for i, data in enumerate(fv_ref):
      ax_time.plot(data['times'], data['residuals'], color='orange', linestyle=mg_linestyles[i%4], label=f'Ref pre')
      if plot_work: ax_work.plot(data['work'], data['residuals'], color='orange')
      # max_time = np.max([max_time, data['times'].max()])

   for i, data in enumerate(p_mg):
      res = data['residuals']
      times = data['times']
      ax_time.plot(times, res,
                  color=mg_colors[data['mg_levels']],
                  linestyle=mg_linestyles[0],
                  label=f'p-MG {data["mg_levels"]} lvl, {data["num_pre_smoothe"]}/{data["num_post_smoothe"]}'
                        f' sm, p-cfl {data["cfl"]}')
      if plot_work:
         work = data['work']
         ax_work.plot(work, res,
                     color=mg_colors[data['mg_levels']],
                     linestyle=mg_linestyles[0])

   for i, data in enumerate(fv_mg):
      # ls = '--'
      ls = smoothings_linestyles[data['num_pre_smoothe'] + data['num_post_smoothe']]
      res = data['residuals']
      times = data['times']
      ax_time.plot(times, res,
                  color=mg_colors[data['mg_levels']],
                  linestyle=ls,
                  label=f'fv-MG {data["mg_levels"]} lvl, {data["num_pre_smoothe"]}/{data["num_post_smoothe"]} sm,'
                        f' p-cfl {data["cfl"]}')
      if plot_work:
         work = data['work']
         ax_work.plot(work, res,
                     color=mg_colors[data['mg_levels']],
                     linestyle=ls)

   ax_time.grid(True)
   ax_time.set_xlabel('Time (s)')
   # ax_time.set_xlim([1.0, max_time])
   ax_time.set_ylabel('Residual')
   ax_time.set_yscale('log')

   if plot_work:
      # ax_work.set_yscale('log')
      ax_work.grid(True)
      ax_work.set_xlim(left = 1.0, right = max_work)
      ax_work.set_xlabel('Work (estimate)')
      ax_work.set_ylabel('Residual')
      ax_work.set_yscale('log')

   fig.suptitle(f'Time (work) to reach accuracy\n{size[0]}x{size[1]}x{size[2]} elem, {size[3]} dt')
   fig.legend(loc='upper right')
   full_filename = f'error_{size[0]}x{size[1]}x{size[2]}_{int(size[3])}.pdf'
   fig.savefig(full_filename)
   plt.close(fig)
   print(f'Saving {full_filename}')

def plot_time_per_step(no_precond, fv_ref, p_mg, fv_mg, prob):
   """
   Plot solve time for each time step of the simulation (average, when it was run multiple times).
   """

   fig, ax = plt.subplots(1, 1)

   for i, data in enumerate(no_precond):
      ax.plot(data['sim_times'] + data['step_dts'], data['time_per_step'] / data['step_dts'],
              color='black', linestyle=main_linestyles[i], marker='.',
              label=f'No precond, {data["time_integrator"]}, tol {data["solver_tol"]}')

   for i, data in enumerate(fv_ref):
      ax.plot(data['sim_times'] + data['step_dts'], data['time_per_step'] / data['step_dts'],
              marker='.',
              color=ref_colors[i], label=f'Ref precond, ptol {data["precond_tol"]:.0e}')

   for i, data in enumerate(p_mg):
      ax.plot(data['sim_times'] + data['step_dts'], data['time_per_step'] / data['step_dts'],
              color=mg_colors[i], linestyle=mg_linestyles[0], marker='.',
              label=f'p-MG prec ({data["mg_smoother"]})')

   for i, data in enumerate(fv_mg):
      # print(f'sim times: {data["sim_times"]}')
      # print(f'time per step: {data["time_per_step"]}')
      # print(f'dts: {data["step_dts"]}')
      ax.plot(data['sim_times'] + data['step_dts'], data['time_per_step'] / data['step_dts'],
              color=mg_colors[i], marker='.',
              label=f'tol {data["solver_tol"]}, FV-MG prec ({data["mg_smoother"]}), dt {data["initial_dt"]}')

   ax.set_ylim(bottom=0)
   # ax.set_xlim(left=0)

   ax.yaxis.grid()

   ax.set_xlabel(f'Problem time (s) (dt = {prob.dt})')
   ax.set_ylabel('Solve time (s/s)')

   fig.suptitle(
      f'Solver time\n{prob.hori}x{prob.vert}x({prob.order}) elem, {prob.dt} dt\n{prob.grid_type}, {prob.equations}',
      fontsize=11)
   fig.legend(fontsize=9)
   full_filename = f'time_per_step_{prob.short_name()}.pdf'
   fig.savefig(full_filename)
   plt.close(fig)
   print(f'Saving {full_filename}')

def plot_it_per_step(no_precond, fv_ref, p_mg, fv_mg, prob):
   """
   Plot solve time for each time step of the simulation (average, when it was run multiple times).
   """

   fig, ax = plt.subplots(1, 1)

   second_max_it = 0
   for i, data in enumerate(no_precond):
      ax.plot(data['step_ids'][:], data['it_per_step'][:],
              color='black', linestyle=main_linestyles[i],
              label=f'No precond, {data["time_integrator"]}, solver tol {data["solver_tol"]}')
      if len(data['time_per_step']) >= 2:
         val = np.partition(data['it_per_step'], -2)[-2]
         second_max_it = max(second_max_it, val)

   for i, data in enumerate(fv_ref):
      ax.plot(data['it_per_step'][:], color=ref_colors[i], label=f'Ref precond, tol {data["precond_tol"]:.0e}')

   for i, data in enumerate(p_mg):
      ax.plot(data['it_per_step'][:], color=mg_colors[i], linestyle=mg_linestyles[0],
              label=f'p-MG precond ({data["mg_smoother"]})')

   for i, data in enumerate(fv_mg):
      ax.plot(data['it_per_step'][:], color=mg_colors[i],
              label=f'solver tol {data["solver_tol"]}, FV-MG precond ({data["mg_smoother"]})')

   if second_max_it > 0: ax.set_ylim(top=second_max_it * 1.04)
   ax.set_ylim(bottom=0)
   ax.set_xlim(left=0)

   ax.yaxis.grid()

   ax.set_xlabel('Time step')
   ax.set_ylabel('Number of iterations')

   fig.suptitle(f'Solver iterations\n'
                + f'{prob.hori}x{prob.vert}x({prob.order}) elem, {prob.dt} dt\n'
                + f'{prob.grid_type}, {prob.equations}',
                fontsize=11)
   fig.legend(fontsize=9)
   full_filename = f'it_per_step_{prob.short_name()}.pdf'
   fig.savefig(full_filename)
   plt.close(fig)
   print(f'Saving {full_filename}')

def main(args):
   """
   Extract data and call appropriate plotting functions.
   """

   if args.split_dt:
      sizes_query = '''
         select distinct grid_type, equations, dg_order, num_elem_h, num_elem_v, initial_dt
         from results_param
      '''
   else:
      sizes_query = '''
         select distinct grid_type, equations, dg_order, num_elem_h, num_elem_v
         from results_param
      '''

   probs = [ProblemDesc(p) for p in list(db_cursor.execute(sizes_query).fetchall())]

   for p in probs:       print(f'{p}')

   for prob in probs:
      no_precond, fv_ref, p_mg, fv_mg = extract_results(prob)
      if args.plot_residual:      plot_residual(no_precond, fv_ref, p_mg, fv_mg, prob)
      if args.plot_time:          plot_time(no_precond, fv_ref, p_mg, fv_mg, prob)
      if args.error_time:         plot_error_time(no_precond, fv_ref, p_mg, fv_mg, prob)
      if args.plot_time_per_step: plot_time_per_step(no_precond, fv_ref, p_mg, fv_mg, prob)
      if args.plot_it_per_step:   plot_it_per_step(no_precond, fv_ref, p_mg, fv_mg, prob)

   # for size in dt_free_sizes:
   #    pass

if __name__ == '__main__':
   import argparse

   parser = argparse.ArgumentParser(description='Plot results from automated preconditioner tests')
   parser.add_argument('results_file', type=str, help='DB file that contains test results')
   parser.add_argument('--plot-residual', action='store_true', help='Plot residual evolution')
   parser.add_argument('--plot-time', action='store_true', help='Plot the time needed to reach target residual')
   parser.add_argument('--error-time', action='store_true',
                       help='Plot time needed to reach a certain error (residual) level')
   parser.add_argument('--plot-time-per-step', action='store_true',
                       help='Plot time for solve per time step (real time over simulation time)')
   parser.add_argument('--plot-it-per-step', action='store_true', help='Plot solver iteration count per time step')
   parser.add_argument('--split-dt', action='store_true',
                       help="Whether to split plots per dt (rather than all dt's on the same plot)")

   parsed_args = parser.parse_args()

   db_connection = sqlite3.connect(parsed_args.results_file)
   db_connection.create_aggregate("stdev", 1, StdevFunc)
   db_cursor = db_connection.cursor()

   main(parsed_args)
