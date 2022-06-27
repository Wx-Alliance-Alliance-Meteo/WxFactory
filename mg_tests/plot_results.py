#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')  # To be able to make plots even without an X server

import matplotlib.pyplot as plt
import numpy as np
import sqlite3

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
            return None
        return np.sqrt(self.S / (self.k-2))

def read_results(filename):
   results = []
   sizes = []
   with open(filename, 'r') as results_file:
      num_skipped = 0

      def get_num_mg_levels(preconditioner, coarsest_level, order):
         if preconditioner == 'fv-mg':
            return int(np.log2(order / coarsest_level))
         elif preconditioner == 'p-mg':
            return order - coarsest_level
         else:
            return 0

      for line in results_file:
         short_line = line.split('#')[0].strip()
         if short_line == '': continue

         items = [x for x in short_line.split(' ') if x != '']

         def get_lin_sol(name):
            if 'fgmres' in name: return 'fgmres'
            if name in ['mg', 'multigrid']: return 'multigrid'
            return 'Big ERROR'

         def get_interp(name):
            if 'l2' in name: return 'l2'
            if 'lag' in name: return 'lag'
            return 'Big ERROR'

         try:
            res_data_raw = [d.split('/') for d in items[18:]]
            res_data = [(float(d[0]), float(d[1]), float(d[2])) for d in res_data_raw]
            data = {
               'order': int(items[0]),
               'num_elements_h': int(items[1]),
               'num_elements_v': int(items[2]),
               'time_step': int(items[3]),
               'linear_solver': get_lin_sol(items[4]),
               'preconditioner': items[5],
               'interp': get_interp(items[6]),
               'precond_tolerance': float(items[7]),
               'coarsest_level': int(items[8]),
               'mg_smoothe_only': True if items[9] == 'True' else False,
               'num_pre_smoothe': int(items[10]),
               'num_post_smoothe': int(items[11]),
               'cfl': float(items[12]),
               'num_solver_it': int(items[14]),
               'solver_time': float(items[15]),
               'solver_flag': int(items[16]),
               'residuals': np.array([r[0] for r in res_data]),
               'times': np.array([r[1] for r in res_data]),
               'work': np.array([r[2] for r in res_data])
               }
            data['mg_levels'] = get_num_mg_levels(data['preconditioner'], data['coarsest_level'], data['order'])
            # print(f'Line: {data}')

            if data['residuals'].size <= 0: raise IndexError

            results.append(data)
            size = (data['order'], data['num_elements_h'], data['num_elements_v'], data['time_step']) 

            if size not in sizes: sizes.append(size)
         except IndexError:
            # Skip lines where the tests were not completed (no iteration/time data)
            num_skipped += 1
            pass

      if num_skipped > 1:
         print(f'Skipped {num_skipped} invalid lines')
      
   return results, sizes

def keep_best_result(result_set, check_param):
   
   if len(result_set) == 0: return []

   if not check_param:
      best_time = np.inf
      best_it   = np.iinfo(0).max
      best_id   = -1

      for i, data in enumerate(result_set):
         num_it = data['num_solver_it']
         time = data['solver_time']
         if num_it >= 1:
            if num_it < best_it or (num_it == best_it and time < best_time):
               best_it   = num_it
               best_time = time
               best_id   = i
         
      return [result_set[best_id]]

   # Gotta check params... Let's do this
   best = {}
   for i, data in enumerate(result_set):
      param_set = (data['mg_levels'], data['mg_smoothe_only'], data['num_pre_smoothe'], data['num_post_smoothe'])
      if param_set not in best:
         if data['num_solver_it'] >= 1: best[param_set] = (data['num_solver_it'], data['solver_time'], i)
      else:
         best_it, best_time, best_id = best[param_set]
         num_it = data['num_solver_it']
         time   = data['solver_time']
         if num_it >= 1:
            if num_it < best_it or (num_it == best_it and time < best_time):
               best[param_set] = (num_it, time, i)
   
   return [result_set[best[k][2]] for k in best]

def extract_results(result, order, num_elem_h, num_elem_v, dt, best_cfl = True):

   # no_precond = [x for x in result
   #    if x['order'] == order
   #    and x['num_elements_h'] == num_elem_h
   #    and x['num_elements_v'] == num_elem_v
   #    and x['time_step'] == dt
   #    and x['preconditioner'] == 'none']

   # fv_ref = [x for x in result
   #    if x['order'] == order
   #    and x['num_elements_h'] == num_elem_h
   #    and x['num_elements_v'] == num_elem_v
   #    and x['time_step'] == dt
   #    and x['preconditioner'] == 'fv'
   #    and x['precond_tolerance'] <= 1e-5]
   # fv_fast = [x for x in result
   #    if x['order'] == order
   #    and x['num_elements_h'] == num_elem_h
   #    and x['num_elements_v'] == num_elem_v
   #    and x['time_step'] == dt
   #    and x['preconditioner'] == 'fv'
   #    and x['precond_tolerance'] >= 1e-1]


   # p_mg_columns = ['linear_solver', 'precond_interp', 'coarsest_mg_order', 'mg_smoothe_only', 'num_pre_smoothe', 'num_post_smoothe']
   # p_mg_condition = 'precond = "p-mg"'

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

      base_query = f'''
         select distinct {{columns}}
         from ({base_subtable_query})
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
      param_query = base_query.format(columns = ', '.join(columns), precond_condition = precond_condition)
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
         set_result['residuals'] = np.array([r[1] for r in residuals])
         set_result['stdevs']    = np.array([r[2] if r[2] is not None else 0.0 for r in residuals])

         results.append(set_result)

      return results

   p_mg_results = get_single_precond_results(
      ['linear_solver', 'precond_interp', 'coarsest_mg_order', 'mg_smoothe_only', 'num_pre_smoothe', 'num_post_smoothe'],
      'precond = "p-mg"'
   )

   fv_mg_results = get_single_precond_results(
      ['linear_solver', 'precond_interp', 'coarsest_mg_order', 'mg_smoothe_only', 'num_pre_smoothe', 'num_post_smoothe'],
      'precond = "fv-mg"'
   )

   # print(f'p_mg_results: {p_mg_results}')

   fv_ref = get_single_precond_results(['linear_solver', 'precond_tol'], 'precond = "fv"')
   # print(f'fv_results: {fv_ref}')

   no_precond = get_single_precond_results(['linear_solver'], 'precond = "none"')
   # print(f'no precond result: {no_precond}')

   # mg_precond_results = {}
   # # residuals = db_cursor.execute(residual_query2, [order, num_elem_h, num_elem_v, dt, 'p-mg']).fetchall()
   # for mg_type in ['fv-mg']:
   #    # ids = db_cursor.execute(run_id_query, [order, num_elem_h, num_elem_v, dt, mg_type])
   #    # id_list = [id[0] for id in ids]
   #    # id_list.append(0)
   #    # print(f'ids: {ids} (list {id_list})')
   #    # residuals = db_cursor.execute(residual_query, [ids, 0]).fetchall()

   #    if order == 4:
   #       mg_precond_results[mg_type] = [x for x in result
   #          if x['order'] == order
   #          and x['num_elements_h'] == num_elem_h
   #          and x['num_elements_v'] == num_elem_v
   #          and x['time_step'] == dt
   #          # and x['coarsest_level'] != 1
   #          and x['preconditioner'] == mg_type]
   #    elif order == 8:
   #       mg_precond_results[mg_type] = [x for x in result
   #          if x['order'] == order
   #          and x['num_elements_h'] == num_elem_h
   #          and x['num_elements_v'] == num_elem_v
   #          and x['time_step'] == dt
   #          and (x['coarsest_level'] == 4 or x['coarsest_level'] == 1)
   #          and x['preconditioner'] == mg_type]
   #    else:
   #       mg_precond_results[mg_type] = [x for x in result
   #          if x['order'] == order
   #          and x['num_elements_h'] == num_elem_h
   #          and x['num_elements_v'] == num_elem_v
   #          and x['time_step'] == dt
   #          and x['preconditioner'] == mg_type]

   #    mg_precond_results[mg_type].sort(key=lambda k:k['num_pre_smoothe'])
   #    mg_precond_results[mg_type].sort(key=lambda k:k['coarsest_level'])

   # if best_cfl:
   #    no_precond = keep_best_result(no_precond, False)
   #    fv_ref     = keep_best_result(fv_ref, False)
   #    fv_fast    = keep_best_result(fv_fast, False)
   #    mg_precond_results['p-mg']  = keep_best_result(mg_precond_results['p-mg'], True)
   #    mg_precond_results['fv-mg'] = keep_best_result(mg_precond_results['fv-mg'], True)

   # res = np.array([r[1] for r in residuals])
   # stdev = np.array([r[2] for r in residuals])
   # print(f'res = {res}')
   # mg_precond_results['p-mg'][0]['residuals'] = res
   # mg_precond_results['p-mg'][0]['stdevs'] = stdev

   # return no_precond, fv_ref, fv_fast, mg_precond_results['p-mg'], mg_precond_results['fv-mg']
   return no_precond, fv_ref,  p_mg_results, fv_mg_results


mg_linestyles = [None, '--', '-.', ':', None, '--', '-.', ':']
# smoothings_linestyles = [':', '-.', '--', None]
smoothings_linestyles = [None, (0, (1, 4)), (0, (1, 2)), (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (5, 3)), (0, (5, 1)), None]
mg_colors = ['blue', 'green', 'purple', 'teal', 'blue', 'green', 'purple']
ref_colors = ['orange', 'magenta']

def plot_iter(result, sizes):

   no_precond = []
   fv_ref     = []
   fv_fast    = []
   p_mg       = []
   fv_mg      = []

   plot_pairs = []
   for size in sizes:
      res = extract_results(result, size[0], size[1], size[2], size[3])
      no_precond.append(res[0])
      fv_ref.append(res[1])
      fv_fast.append(res[2])
      p_mg.append(res[3])
      fv_mg.append(res[4])

      pair = (size[0], size[3])
      if not pair in plot_pairs: plot_pairs.append(pair)

   no_precond_sets = {}
   fv_ref_sets = {}
   fv_fast_sets = {}
   fv_sets = {}
   p_sets = {}
   for pair in plot_pairs:
      no_precond_sets[pair] = []
      fv_ref_sets[pair]     = []
      fv_fast_sets[pair]    = []
      fv_sets[pair]         = {}
      p_sets[pair]          = {}

   for result_list in no_precond:
      if len(result_list) > 0:
         data = result_list[0]
         size = (data['order'], data['num_elements_h'], data['num_elements_v'], data['time_step'])
         pair = (size[0], size[3])
         no_precond_sets[pair].append(((size[1], size[2]), data['num_solver_it']))

   for result_list in fv_mg:
      for data in result_list:
         size = (data['order'], data['num_elements_h'], data['num_elements_v'], data['time_step'])
         pair = (size[0], size[3])
         param_set = (data['mg_levels'], data['mg_smoothe_only'], data['num_pre_smoothe'], data['num_post_smoothe'])
         if param_set not in fv_sets[pair]: fv_sets[pair][param_set] = []
         fv_sets[pair][param_set].append(((size[1], size[2]), data['num_solver_it']))
         # print(f'fv_sets[{pair}][{param_set}] = {fv_sets[pair][param_set]}')

   for result_list in p_mg:
      for data in result_list:
         size = (data['order'], data['num_elements_h'], data['num_elements_v'], data['time_step'])
         param_set = (data['mg_levels'], data['mg_smoothe_only'], data['num_pre_smoothe'], data['num_post_smoothe'])
         pair = (size[0], size[3])
         if param_set not in p_sets[pair]: p_sets[pair][param_set] = []
         p_sets[pair][param_set].append(((size[1], size[2]), data['num_solver_it']))

   for pair in plot_pairs:
      
      sizes = []
      for res in no_precond_sets[pair]:
         if res[0] not in sizes: sizes.append(res[0])
      
      sizes.sort()
      
      bar_sets = {}
      for size in sizes:
         bar_sets[size] = []

      labels = ['No precond']
      for res in no_precond_sets[pair]:
         size = res[0]
         bar_sets[size].append(res[1])

      for param_set in fv_sets[pair]:
         labels.append(f'FV {param_set[0]} lvl, {param_set[2]}/{param_set[3]}')
         for res in fv_sets[pair][param_set]:
            size = res[0]
            # print(f'entire row: {fv_sets[pair][param_set]}')
            # print(f'pair: {pair}')
            # print(f'param_set  {param_set}')
            # print(f'res = {res}')
            # print(f'size = {size}')
            bar_sets[size].append(res[1])

      num_sizes = len(bar_sets)
      num_blobs = 0
      width = 0.9 / num_sizes

      fig, ax_it = plt.subplots()
      for i, size in enumerate(bar_sets):
         num_blobs = len(data)
         data = bar_sets[size]
         ax_it.bar(np.arange(len(data)) + width*(i-num_sizes/2+1.5), data, width=width, label=f'{size[0]}x{size[1]}')

      ax_it.set_xticks(np.arange(num_blobs) + width)
      ax_it.set_xticklabels(labels)
      plt.setp(ax_it.get_xticklabels(), rotation=20, ha="right",
         rotation_mode="anchor")

      fig.suptitle(f'Iteration count\norder {pair[0]}, dt = {pair[1]}')
      fig.legend()
      full_filename = f'iteration_{pair[0]}_{pair[1]}.pdf'
      fig.savefig(full_filename)
      plt.close(fig)
      print(f'Saving {full_filename}')
      # break

def plot_residual(result, order, num_elem_h, num_elem_v, dt):

   no_precond, fv_ref, p_mg, fv_mg = extract_results(result, order, num_elem_h, num_elem_v, dt, False)
   MAX_IT = 20
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

   # for i, data in enumerate(fv_fast):
   #    ax_res.plot(data['residuals'][:MAX_IT], color='magenta', linestyle=mg_linestyles[i%4], label=f'Simple FV precond')
   #    max_res = max(data['residuals'][:MAX_IT].max(), max_res)
   #    min_res = min(data['residuals'][:MAX_IT].min(), min_res)
   #    break

   for i, data in enumerate(p_mg):
      # ax_res.plot(data['residuals'][:MAX_IT], color=mg_colors[data['mg_levels']],
      ax_res.errorbar(x = np.arange(len(data['residuals'][:MAX_IT])),
            y = data['residuals'][:MAX_IT], yerr = data['stdevs'][:MAX_IT],
            color=mg_colors[data['coarsest_mg_order']], linestyle=mg_linestyles[0],
            label=f'p-MG prec ({data["coarsest_mg_order"]} lvl, {data["num_pre_smoothe"]}/{data["num_post_smoothe"]} sm)')
      max_res = max(data['residuals'][:MAX_IT].max(), max_res)
      min_res = min(data['residuals'][:MAX_IT].min(), min_res)

   for i, data in enumerate(fv_mg):
      ls = smoothings_linestyles[data['num_pre_smoothe'] + data['num_post_smoothe']]
      # if data['smoother'] == 'irk': ls = ':'
      ax_res.errorbar(
            np.arange(len(data['residuals'][:MAX_IT])),
            data['residuals'][:MAX_IT], yerr = data['stdevs'][:MAX_IT],
            color=mg_colors[data['coarsest_mg_order']], linestyle=ls,
            label=f'FV-MG prec {data["coarsest_mg_order"]} lvl, {data["num_pre_smoothe"]}/{data["num_post_smoothe"]} sm')
      max_res = max(data['residuals'][:MAX_IT].max(), max_res)
      min_res = min(data['residuals'][:MAX_IT].min(), min_res)

   ax_res.set_xlabel('Iteration #')
   if max_it > 0: ax_res.set_xlim([-1, min(max_it, MAX_IT)])
   # ax_res.set_xscale('log')

   ax_res.set_yscale('log')
   ax_res.grid(True)
   ax_res.set_ylabel('Residual')
   if max_res > 0.0: ax_res.set_ylim([min_res * 0.8, min(2e0, max_res * 1.5)])

   fig.suptitle(f'Residual evolution\n{order}x{num_elem_h}x{num_elem_v} elem, {dt} dt')
   fig.legend()
   full_filename = f'residual_{order}x{num_elem_h}x{num_elem_v}_{int(dt)}.pdf'
   fig.savefig(full_filename)
   plt.close(fig)
   print(f'Saving {full_filename}')

def plot_error_time(result, order, num_elem_h, num_elem_v, dt):

   no_precond, fv_ref, fv_fast, p_mg, fv_mg = extract_results(result, order, num_elem_h, num_elem_v, dt)

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

   for i, data in enumerate(fv_fast):
      ax_time.plot(data['times'], data['residuals'], color='magenta', linestyle=mg_linestyles[i%4], label=f'Simple FV pre')
      # max_time = np.max([max_time, data['times'].max()])
      if plot_work: ax_work.plot(data['work'], data['residuals'], color='magenta', linestyle=mg_linestyles[i%4])

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
      # if data['smoother'] == 'irk': ls = ':'
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

   fig.suptitle(f'Time (work) to reach accuracy\n{order}x{num_elem_h}x{num_elem_v} elem, {dt} dt')
   fig.legend(loc='upper right')
   full_filename = f'error_{order}x{num_elem_h}x{num_elem_v}_{int(dt)}.pdf'
   fig.savefig(full_filename)
   plt.close(fig)
   print(f'Saving {full_filename}')

def main(args):
   # results, sizes = read_results(args.results_file)
   results = None

   sizes_query = '''
      select distinct dg_order, num_elem_h, num_elem_v, dt
      from results_param
   '''
   sizes = [s for s in db_cursor.execute(sizes_query).fetchall()]

   print(f'Sizes: {sizes}')

   if args.plot_iter:
      plot_iter(results, sizes)

   if args.plot_residual:
      for size in sizes:
         plot_residual(results, size[0], size[1], size[2], size[3])

   if args.error_time:
      for size in sizes:
         plot_error_time(results, size[0], size[1], size[2], size[3])

if __name__ == '__main__':
   import argparse

   parser = argparse.ArgumentParser(description='Plot results from automated preconditioner tests')
   parser.add_argument('results_file', type=str, help='File that contains test results')
   parser.add_argument('--db', type=str, help='Database file containing the results to plot')
   parser.add_argument('--plot-iter', action='store_true', help='Plot the iterations and time with respect to various parameters')
   parser.add_argument('--plot-residual', action='store_true', help='Plot residual evolution')
   parser.add_argument('--error-time', action='store_true', help='Plot time needed to reach a certain error (residual) level')

   args = parser.parse_args()

   if args.db is not None:
      db_connection = sqlite3.connect(args.db)
      db_connection.create_aggregate("stdev", 1, StdevFunc)
      db_cursor     = db_connection.cursor()

   main(args)
