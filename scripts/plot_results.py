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

def extract_results(order: int, num_elem_h: int, num_elem_v: int, dt: int) \
      -> Tuple[SingleResultSet, SingleResultSet, SingleResultSet, SingleResultSet]:
   """
   Extract the content of the open database into 4 sets: "no precond", "fv precond", "p-MG precond", "FV-MG" precond.
   """

   def get_single_precond_results(columns, precond_condition) \
         -> SingleResultSet:
      """
      Arguments:
      columns           -- Which additional columns you want to have in the set of results
      precond_condition -- A custom condition to be included in the query
      """

      # Query to select a subtable containing all columns, for only the problems that have the specified
      # size/order/timestep and the specified custom condition
      base_subtable_query = f'''
         select * from results_param
         where
            dg_order   = {order} AND
            num_elem_h = {num_elem_h} AND
            num_elem_v = {num_elem_v} AND
            initial_dt = {dt} AND
            {{precond_condition}}
      '''

      time_per_step_query = f'''
         select *, avg(total_solve_time) as step_time, avg(num_solver_it) as step_it
         from ({base_subtable_query})
         group by {{columns}}, step_id
         order by num_mg_levels, mg_solve_coarsest, kiops_dt_factor, (num_pre_smoothe + num_post_smoothe), step_id
      '''.strip().format(columns = ', '.join(columns), precond_condition = precond_condition)

      # Query to compute statistics from multiple time steps for each set of precondtioner parameters that
      # was found in the subtable (according to the given columns)
      base_param_query = f'''
         select {{columns}}, group_concat(step_id), group_concat(step_time), avg(step_time), stdev(step_time), avg(step_it), stdev(step_it)
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
      param_query = base_param_query.format(columns = ', '.join(columns), precond_condition = precond_condition)
      param_sets = db_cursor.execute(param_query).fetchall()

      for subset in param_sets:
         # print(f'subset = {subset}')
         residual_query = base_residual_query.format(
            inner_condition = ' AND '.join([f'{c} = "{subset[i]}"' for i, c in enumerate(columns)]),
            precond_condition = precond_condition)
         # print(f'res query: {residual_query}')
         residuals = db_cursor.execute(residual_query).fetchall()

         set_result = {}
         for i, c in enumerate(columns):
            set_result[c] = subset[i]
         set_result['step_ids']      = np.array([int(x) for x in subset[-6].split(',')])
         set_result['time_per_step'] = np.array([float(x) for x in subset[-5].split(',')])
         set_result['time_avg']      = subset[-4]
         set_result['time_stdev']    = subset[-3]
         set_result['it_avg']        = subset[-2]
         set_result['it_stdev']      = subset[-1]
         set_result['residuals']     = np.array([r[1] for r in residuals])
         set_result['stdevs']        = np.array([r[2] if r[2] is not None else 0.0 for r in residuals])

         results.append(set_result)

         # print(f'time per step: {set_result["time_per_step"]}')

      return results

   t0 = time()
   no_precond = get_single_precond_results(['time_integrator', 'solver_tol'], 'precond = "none"')

   # no_precond_time = no_precond[0]['time_avg']
   # print(f'precond avg = {no_precond_time}')

   t1 = time()
   fv_ref = get_single_precond_results(['solver_tol', 'precond_tol'], f'precond = "fv" AND precond_tol > 1e-2')

   t2 = time()
   p_mg_results = get_single_precond_results(
      ['solver_tol', 'precond_interp', 'num_mg_levels', 'kiops_dt_factor', 'mg_solve_coarsest', 'precond_tol',
       'num_pre_smoothe', 'num_post_smoothe', 'mg_smoother'],
      f'precond = "p-mg"'
   )

   t3 = time()
   fv_mg_results = get_single_precond_results(
      ['solver_tol', 'precond_interp', 'num_mg_levels', 'kiops_dt_factor', 'mg_solve_coarsest', 'precond_tol',
       'num_pre_smoothe', 'num_post_smoothe', 'mg_smoother'],
      f'precond = "fv-mg"'
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

def plot_residual(no_precond, fv_ref, p_mg, fv_mg, size):
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

def plot_time(no_precond, fv_ref, p_mg, fv_mg, size):
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

def plot_error_time(no_precond, fv_ref, p_mg, fv_mg, size):
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

def plot_time_per_step(no_precond, fv_ref, p_mg, fv_mg, size):
   """
   Plot solve time for each time step of the simulation (average, when it was run multiple times).
   """

   fig, ax = plt.subplots(1, 1)

   for i, data in enumerate(no_precond):
      ax.plot(data['step_ids'][:], data['time_per_step'][:],
              color='black', linestyle=main_linestyles[i],
              label=f'No precond, {data["time_integrator"]}, solver tol {data["solver_tol"]}')

   for i, data in enumerate(fv_ref):
      ax.plot(data['time_per_step'][:], color=ref_colors[i], label=f'Ref precond, tol {data["precond_tol"]:.0e}')

   for i, data in enumerate(p_mg):
      ax.plot(data['time_per_step'][:], color=mg_colors[i], linestyle=mg_linestyles[0],
              label=f'p-MG precond ({data["mg_smoother"]})')

   for i, data in enumerate(fv_mg):
      ax.plot(data['time_per_step'][:], color=mg_colors[i],
              label=f'solver tol {data["solver_tol"]}, FV-MG precond ({data["mg_smoother"]})')

   timestamp_no_prec = {}
   timestamp_no_prec[(4, 19, 30, 5)] = [
      4049.58, 4096.46, 4141.59, 4183.91, 4223.18, 4260.76, 4294.89, 4327.06, 4357.97, 4386.33,
      4412.96, 4436.62, 4461.66, 4488.08, 4513.49, 4538.87, 4563.30, 4587.79, 4612.27, 4636.52,
      4660.41, 4683.24, 4704.33, 4725.30, 4745.86, 4765.62, 4784.42, 4801.55, 4818.16, 4834.54,
      4850.96, 4867.15, 4883.11, 4898.21, 4913.73, 4929.06, 4944.20, 4960.05, 4975.64, 4991.69,
      5007.95, 5023.95, 5040.44, 5057.02, 5073.74, 5090.71, 5107.58, 5124.70, 5142.00, 5159.17,
      5176.72
   ]
   timestamp_no_prec[(4, 39, 60, 5)] = [
      794630.47, 795152.21, 795641.06, 796102.44, 796538.85, 796947.38, 797333.68, 797701.45, 798057.35, 798408.56,
      798760.71, 799110.18, 799453.46, 799789.50, 800107.99, 800420.89, 800732.14, 801039.37, 801331.66, 801609.23,
      801883.11, 802136.05, 802403.68, 802649.28, 802889.05, 803129.81, 803364.42, 803598.43, 803819.92, 804051.40,
      804280.58, 804516.77, 804745.59, 804987.31, 805181.94, 805391.00, 805592.30, 805806.03, 806016.52, 806226.13,
      806445.91, 806661.02, 806893.46, 807134.66, 807345.00, 807570.94, 807805.72, 808038.06, 808270.06, 808503.09,
      808739.16,
   ]

   timestamp_prec = {}
   timestamp_prec[(4, 19, 30, 5)] = [
      4157.98, 4169.60, 4181.72, 4193.32, 4205.44, 4217.05, 4228.66, 4240.33, 4251.95, 4263.55,
      4275.16, 4286.92, 4298.84, 4310.75, 4323.02, 4335.29, 4347.72, 4359.63, 4371.55, 4383.62,
      4395.85, 4408.08, 4420.30, 4433.55, 4446.95, 4460.50, 4474.06, 4487.61, 4501.16, 4514.88,
      4528.74, 4542.60, 4556.46, 4570.63, 4584.80, 4598.98, 4613.15, 4627.47, 4641.95, 4656.58,
      4671.53, 4686.62, 4701.72, 4716.81, 4732.06, 4747.46, 4763.03, 4778.74, 4794.61, 4810.63,
      4826.81,
   ]
   timestamp_prec[(4, 39, 60, 5)] = [
      4540.37, 4616.68, 4692.99, 4765.09, 4835.33, 4905.00, 4975.86, 5045.52, 5113.96, 5186.63,
      5258.71, 5330.76, 5403.46, 5477.96, 5552.47, 5627.58, 5703.28, 5780.30, 5857.25, 5935.39,
      6013.55, 6091.71, 6169.87, 6247.44, 6325.15, 6402.23, 6479.77, 6556.68, 6633.60, 6709.91,
      6786.22, 6862.53, 6938.24, 7014.54, 7089.63, 7165.33, 7241.64, 7317.93, 7394.27, 7472.44,
      7550.57, 7630.55, 7711.14, 7792.33, 7874.15, 7955.36, 8037.79, 8120.23, 8202.67, 8285.11,
      8366.93,
   ]

   time_no_prec = {}
   time_prec = {}

   for key, item in timestamp_no_prec.items():
      time_no_prec[key] = [float(item[i+1]) - float(item[i]) for i in range(len(item) - 1)]

   for key, item in timestamp_prec.items():
      time_prec[key] = [float(item[i+1]) - float(item[i]) for i in range(len(item) - 1)]

   if size in time_no_prec:
      ax.plot(time_no_prec[size], color='black', label=f'Dune no precond', linestyle='--')

   if size in time_prec:
      ax.plot(time_prec[size], color=mg_colors[2], label=f'Dune precond', linestyle='--')

   _, t = ax.get_ylim()
   top_limit = size[0] * size[1] * size[2] * 0.8

   ax.set_ylim(bottom=0, top=min(top_limit, t))
   ax.set_xlim(left=0)

   ax.set_xlabel('Time step')
   ax.set_ylabel('Solve time (s)')

   fig.suptitle(f'Solver time\n{size[0]}x{size[1]}x{size[2]} elem, {size[3]} dt')
   fig.legend()
   full_filename = f'time_per_step_{size[0]}x{size[1]}x{size[2]}_{int(size[3])}.pdf'
   fig.savefig(full_filename)
   plt.close(fig)
   print(f'Saving {full_filename}')

def main(args):
   """
   Extract data and call appropriate plotting functions.
   """

   sizes_query = '''
      select distinct dg_order, num_elem_h, num_elem_v, initial_dt
      from results_param
   '''
   sizes = list(db_cursor.execute(sizes_query).fetchall())
   dt_independent_size_query = '''
      select distinct dg_order, num_elem_h, num_elem_v
      from results_param
   '''
   dt_free_sizes = list(db_cursor.execute(dt_independent_size_query).fetchall())

   print(f'Sizes: {sizes}, {dt_free_sizes}')

   for size in sizes:
      no_precond, fv_ref, p_mg, fv_mg = extract_results(size[0], size[1], size[2], size[3])
      if args.plot_residual:      plot_residual(no_precond, fv_ref, p_mg, fv_mg, size)
      if args.plot_time:          plot_time(no_precond, fv_ref, p_mg, fv_mg, size)
      if args.error_time:         plot_error_time(no_precond, fv_ref, p_mg, fv_mg, size)
      if args.plot_time_per_step: plot_time_per_step(no_precond, fv_ref, p_mg, fv_mg, size)

   for size in sizes:
      pass

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

   parsed_args = parser.parse_args()

   db_connection = sqlite3.connect(parsed_args.results_file)
   db_connection.create_aggregate("stdev", 1, StdevFunc)
   db_cursor = db_connection.cursor()

   main(parsed_args)
