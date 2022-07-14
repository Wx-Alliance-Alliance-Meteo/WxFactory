#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')  # To be able to make plots even without an X server

import matplotlib.pyplot as plt
import numpy as np
import sqlite3
from time import time

db_connection = None
db_cursor     = None

class StdevFunc:
    def __init__(self):
        self.M = 0.0
        self.S = 0.0
        self.k = 1

    def step(self, value):
        if value is None:
            return
        tM = self.M
        self.M += (value - tM) / self.k
        self.S += (value - tM) * (value - self.M)
        self.k += 1

    def finalize(self):
        if self.k < 3:
            return 0.0
        return np.sqrt(self.S / (self.k-2))

def extract_results(order, num_elem_h, num_elem_v, dt):

   def get_single_precond_results(columns, precond_condition):
      base_subtable_query = f'''
         select * from results_param
         where
            dg_order   = {order} AND
            num_elem_h = {num_elem_h} AND
            num_elem_v = {num_elem_v} AND
            dt         = {dt} AND
            {{precond_condition}}
      '''

      base_param_query = f'''
         select {{columns}}, avg(solver_time), stdev(solver_time), avg(num_solver_it), stdev(num_solver_it)
         from ({base_subtable_query})
         group by {{columns}}
         order by num_mg_levels, mg_smoothe_only, kiops_dt_factor, (num_pre_smoothe + num_post_smoothe)
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
      
      for set in param_sets:
         # print(f'set = {set}')
         residual_query = base_residual_query.format(
            inner_condition = ' AND '.join([f'{c} = "{set[i]}"' for i, c in enumerate(columns)]),
            precond_condition = precond_condition)
         # print(f'res query: {residual_query}')
         residuals = db_cursor.execute(residual_query).fetchall()

         set_result = {}
         for i, c in enumerate(columns):
            set_result[c] = set[i]
         set_result['time_avg']   = set[-4]
         set_result['time_stdev'] = set[-3]
         set_result['it_avg']     = set[-2]
         set_result['it_stdev']   = set[-1]
         set_result['residuals']  = np.array([r[1] for r in residuals])
         set_result['stdevs']     = np.array([r[2] if r[2] is not None else 0.0 for r in residuals])

         results.append(set_result)

      return results

   t0 = time()
   no_precond = get_single_precond_results(['linear_solver'], 'precond = "none"')

   no_precond_time = no_precond[0]['time_avg']
   print(f'precond avg = {no_precond_time}')

   t1 = time()
   fv_ref = get_single_precond_results(['linear_solver', 'precond_tol'], f'precond = "fv" AND precond_tol > 1e-2')

   t2 = time()
   p_mg_results = get_single_precond_results(
      ['linear_solver', 'precond_interp', 'num_mg_levels', 'kiops_dt_factor', 'mg_smoothe_only', 'precond_tol', 'num_pre_smoothe', 'num_post_smoothe'],
      f'precond = "p-mg" AND solver_time < {3.0 * no_precond_time} AND (mg_smoothe_only = 1 OR precond_tol > 1e-5)'
   )

   t3 = time()
   fv_mg_results = get_single_precond_results(
      ['linear_solver', 'precond_interp', 'num_mg_levels', 'kiops_dt_factor', 'mg_smoothe_only', 'precond_tol', 'num_pre_smoothe', 'num_post_smoothe'],
      f'precond = "fv-mg" AND solver_time < {3.0 * no_precond_time} AND (mg_smoothe_only = 1 OR precond_tol > 1e-5)'
   )

   t4 = time()

   print(f'extracted results in {t4 - t0:.2f}s ({t1 - t0:.3f}, {t2 - t1:.3f}, {t3 - t2:.3f}, {t4 - t3:.3f})')

   return no_precond, fv_ref, p_mg_results, fv_mg_results


mg_linestyles = [None, '--', '-.', ':', None, '--', '-.', ':']
# smoothings_linestyles = [':', '-.', '--', None]
smoothings_linestyles = [None, (0, (1, 4)), (0, (1, 2)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (5, 3)), (0, (5, 1)), None]
mg_colors = ['blue', 'green', 'purple', 'teal', 'blue', 'green', 'purple']
ref_colors = ['orange', 'magenta']

def plot_residual(no_precond, fv_ref, p_mg, fv_mg, size):

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
   if max_it > 0: ax_res.set_xlim([-1, min(max_it, MAX_IT)])
   # ax_res.set_xscale('log')

   ax_res.set_yscale('log')
   ax_res.grid(True)
   ax_res.set_ylabel('Residual')
   if max_res > 0.0: ax_res.set_ylim([max(min_res * 0.8, 5e-8), min(2e0, max_res * 1.5)])

   fig.suptitle(f'Residual evolution\n{size[0]}x{size[1]}x{size[2]} elem, {size[3]} dt')
   fig.legend()
   full_filename = f'residual_{size[0]}x{size[1]}x{size[2]}_{int(size[3])}.pdf'
   fig.savefig(full_filename)
   plt.close(fig)
   print(f'Saving {full_filename}')

def plot_time(no_precond, fv_ref, p_mg, fv_mg, size):

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
            names.append(f'p-MG, {data["num_mg_levels"]} lvl, {data["num_pre_smoothe"]}/{data["num_post_smoothe"]} sm, {data["mg_smoothe_only"]}, {data["precond_tol"]:.0e}')
            times.append(data['time_avg'])
            errors.append(data['time_stdev'])

      for i, data in enumerate(fv_mg):
         pass

      ax_time.barh(names, times, xerr=errors, log=False)
      ax_time.invert_yaxis()

      fig.suptitle(f'Time to solution for each timestep\n{size[0]}x{size[1]}x{size[2]} elem, {size[3]} dt\nkiops factor {factor}')
      full_filename = f'step_time_{size[0]}x{size[1]}x{size[2]}_{int(size[3])}_{int(factor*100):03d}.pdf'
      fig.savefig(full_filename)
      plt.close(fig)
      print(f'Saving {full_filename}')
   

def plot_error_time(no_precond, fv_ref, p_mg, fv_mg, size):

   max_time  = 0.0
   max_work  = 0.0
   plot_work = False
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
      time = data['times']
      ax_time.plot(time, res,
                  color=mg_colors[data['mg_levels']],
                  linestyle=mg_linestyles[0],
                  label=f'p-MG {data["mg_levels"]} lvl, {data["num_pre_smoothe"]}/{data["num_post_smoothe"]} sm, p-cfl {data["cfl"]}')
      if plot_work:
         work = data['work']
         ax_work.plot(work, res, 
                     color=mg_colors[data['mg_levels']],
                     linestyle=mg_linestyles[0])

   for i, data in enumerate(fv_mg):
      # ls = '--'
      ls = smoothings_linestyles[data['num_pre_smoothe'] + data['num_post_smoothe']]
      res = data['residuals']
      time = data['times']
      ax_time.plot(time, res,
                  color=mg_colors[data['mg_levels']],
                  linestyle=ls,
                  label=f'fv-MG {data["mg_levels"]} lvl, {data["num_pre_smoothe"]}/{data["num_post_smoothe"]} sm, p-cfl {data["cfl"]}')
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
      ax_work.set_xlim([1.0, max_work])
      ax_work.set_xlabel('Work (estimate)')
      ax_work.set_ylabel('Residual')
      ax_work.set_yscale('log')

   fig.suptitle(f'Time (work) to reach accuracy\n{size[0]}x{size[1]}x{size[2]} elem, {size[3]} dt')
   fig.legend(loc='upper right')
   full_filename = f'error_{size[0]}x{size[1]}x{size[2]}_{int(size[3])}.pdf'
   fig.savefig(full_filename)
   plt.close(fig)
   print(f'Saving {full_filename}')

def main(args):
   sizes_query = '''
      select distinct dg_order, num_elem_h, num_elem_v, dt
      from results_param
   '''
   sizes = [s for s in db_cursor.execute(sizes_query).fetchall()]

   print(f'Sizes: {sizes}')

   for size in sizes:
      no_precond, fv_ref, p_mg, fv_mg = extract_results(size[0], size[1], size[2], size[3])
      if args.plot_residual: plot_residual(no_precond, fv_ref, p_mg, fv_mg, size)
      if args.plot_time:     plot_time(no_precond, fv_ref, p_mg, fv_mg, size)
      if args.error_time:    plot_error_time(no_precond, fv_ref, p_mg, fv_mg, size)

if __name__ == '__main__':
   import argparse

   parser = argparse.ArgumentParser(description='Plot results from automated preconditioner tests')
   parser.add_argument('results_file', type=str, help='DB file that contains test results')
   parser.add_argument('--plot-residual', action='store_true', help='Plot residual evolution')
   parser.add_argument('--plot-time', action='store_true', help='Plot the time needed to reach target residual')
   parser.add_argument('--error-time', action='store_true', help='Plot time needed to reach a certain error (residual) level')

   args = parser.parse_args()

   db_connection = sqlite3.connect(args.results_file)
   db_connection.create_aggregate("stdev", 1, StdevFunc)
   db_cursor = db_connection.cursor()

   main(args)
