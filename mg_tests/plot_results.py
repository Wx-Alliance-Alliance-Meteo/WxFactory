#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def read_results(filename):
   results = []
   with open(filename, 'r') as results_file:
      for line in results_file:
         short_line = line.split('#')[0].strip()
         if short_line == '': continue

         items = [x for x in short_line.split(' ') if x != '']

         def get_lin_sol(name):
            if 'fgmres' in name: return 'fgmres'
            if name in ['mg', 'multigrid']: return 'multigrid'
            return 'Big ERROR'

         def get_precond(num):
            if num == 0: return 'None'
            if num == 1: return 'Finite volume'
            if num == 2: return 'Multigrid'
            return 'Big ERROR'

         def get_interp(name):
            if 'l2' in name: return 'l2'
            if 'lag' in name: return 'lag'
            return 'Big ERROR'

         try:
            data = {
               'order': int(items[0]),
               'num_elements': int(items[1]),
               'time_step': int(items[2]),
               'linear_solver': get_lin_sol(items[3]),
               'preconditioner': get_precond(int(items[4])),
               'interp': get_interp(items[5]),
               'precond_tolerance': float(items[6]),
               'mg_level': int(items[7]),
               'mg_smoothe_only': True if int(items[8]) == 1 else False,
               'num_pre_smoothe': int(items[9]),
               'num_post_smoothe': int(items[10]),
               'cfl': float(items[11]),
               'num_fgmres_it': int(items[13]),
               'fgmres_time': float(items[14]),
               'num_precond_it': int(items[15]),
               'precond_time': float(items[16]),
               'return_flag': int(items[17]),
               'residuals': [float(x) for x in items[19:]]
               }
            # print(f'Line: {data}')
            results.append(data)
         except IndexError:
            # Skip lines where the tests were not completed (no iteration/time data)
            pass
   return results

def get_plot_data(results, order, precond, tol=None):

   if tol is None:
      res = [x for x in results if x['order'] == order and x['preconditioner'] == precond]
   else:
      res = [x for x in results if x['order'] == order and x['preconditioner'] == precond and x['precond_tolerance'] == tol]

   all_data = [3 * ((x['num_elements']*order)**2, x['num_fgmres_it'], x['fgmres_time']) for x in res]
   all_data.sort()
   sizes = [x[0] for x in all_data]
   it = [x[1] for x in all_data]
   times = [x[2] for x in all_data]

   return sizes, it, times

colors = ['blue', 'orange', 'green']

def plot_base_results(results, filename):
   # fig, (ax_it, ax_time) = plt.subplots(1, 2, figsize=(9,4))
   fig, (ax_it, ax_time) = plt.subplots(2, 1)

   for i, order in enumerate([2, 4, 8]):
      sizes, it, times = get_plot_data(results, order, 'None')
      ax_it.plot(sizes, it, 'o-', color=colors[i], label=f'None, o{order}')
      ax_time.plot(sizes, times, 'o-', color=colors[i])

   for i, order in enumerate([2, 4, 8]):
      sizes, it, times = get_plot_data(results, order, 'Finite volume', tol=1e-7)
      ax_it.plot(sizes, it, 'x-', color=colors[i], label=f'FV, o{order}')
      ax_time.plot(sizes, times, 'x-', color=colors[i])

   # ax_it.set_xlabel('# DOFs')
   ax_it.set_ylabel('# Iterations')
   ax_time.set_xlabel('# DOFs')
   ax_time.set_ylabel('Time (s)')

   ax_it.set_xscale('log')
   ax_it.set_yscale('log')
   ax_time.set_xscale('log')
   ax_time.set_yscale('log')

   fig.suptitle('No precond vs reference preconditioner')
   fig.legend(loc='lower right')
   fig.tight_layout()
   fig.savefig(filename)

def plot_fv(results, filename):
   fig, (ax_it, ax_time) = plt.subplots(2, 1)

   for i, order in enumerate([2, 4, 8]):
      sizes, it, times = get_plot_data(results, order, 'Finite volume', tol=1e-7)
      ax_it.plot(sizes, it, 'o-', color=colors[i], label=f'FV ref, o{order}')
      ax_time.plot(sizes, times, 'o-', color=colors[i])

   for i, order in enumerate([2, 4, 8]):
      sizes, it, times = get_plot_data(results, order, 'Finite volume', tol=1e-1)
      ax_it.plot(sizes, it, 'x-', color=colors[i], label=f'FV fast, o{order}')
      ax_time.plot(sizes, times, 'x-', color=colors[i])

   ax_it.set_ylabel('# Iterations')
   ax_time.set_xlabel('# DOFs')
   ax_time.set_ylabel('Time (s)')

   ax_it.set_xscale('log')
   ax_it.set_yscale('log')
   ax_time.set_xscale('log')
   ax_time.set_yscale('log')

   fig.suptitle('Reference FV vs Fast FV')
   fig.legend(loc='lower right')
   fig.tight_layout()
   fig.savefig(filename)

def plot_residual(result, order, num_elem, filename):
   no_precond = [x['residuals'] for x in result
      if x['order'] == order
      and x['num_elements'] == num_elem
      and x['preconditioner'] == 'None'][0][:min(order*num_elem, 100)]
   fv_ref = [x['residuals'] for x in result
      if x['order'] == order
      and x['num_elements'] == num_elem
      and x['preconditioner'] == 'Finite volume'
      and x['precond_tolerance'] == 1e-7]
   fv_fast = [x['residuals'] for x in result
      if x['order'] == order
      and x['num_elements'] == num_elem
      and x['preconditioner'] == 'Finite volume'
      and x['precond_tolerance'] == 1e-1]

   multigrid_results = [x for x in result
      if x['order'] == order
      and x['num_elements'] == num_elem
      and x['preconditioner'] == 'Multigrid']

   fig, ax_res = plt.subplots(1, 1)

   ax_res.plot(no_precond, label='No precond')
   for data in fv_ref:
      ax_res.plot(data, label='Reference precond')
   for data in fv_fast:
      ax_res.plot(data, label='Fast FV precond')

   # mg_linestyles = {20: '-.', 50: ':', 100: '--', 200: '-.', 300: ':', 400: '--'}
   # mg_colors = ['black', 'blue', 'green', 'orange']
   mg_linestyles = [None, '--', '-.', ':', None, '--', '-.', ':']
   mg_colors = {20: 'blue', 50: 'green', 100: 'purple', 200: 'blue', 300: 'green', 400: 'purple'}
   for data in multigrid_results:
      res = data['residuals']
      ax_res.plot(res,
                  linestyle=mg_linestyles[data['num_pre_smoothe']],
                  label=f'sm={data["num_pre_smoothe"]}')

   ax_res.set_yscale('log')
   ax_res.grid(True)

   fig.legend()
   fig.savefig(f'{filename}_{order}x{num_elem}.png')

def plot_residual_2(result, order, num_elem, filename):

   MAX_IT = 80

   no_precond = [x['residuals'] for x in result
      if x['order'] == order
      and x['num_elements'] == num_elem
      and x['preconditioner'] == 'None'
      and x['linear_solver'] == 'fgmres']
   fv_ref = [x for x in result
      if x['order'] == order
      and x['num_elements'] == num_elem
      and x['preconditioner'] == 'Finite volume'
      and x['precond_tolerance'] == 1e-7]
   fv_fast = [x for x in result
      if x['order'] == order
      and x['num_elements'] == num_elem
      and x['preconditioner'] == 'Finite volume'
      and x['precond_tolerance'] == 1e-1]
   mg_solver = [x for x in result
      if x['order'] == order
      and x['num_elements'] == num_elem
      and x['preconditioner'] == 'None'
      and x['linear_solver'] == 'multigrid']

   mg_solver.sort(key=lambda k:k['num_pre_smoothe'])
   mg_solver.sort(key=lambda k:k['mg_level'])

   if order == 4:
      multigrid_results = [x for x in result
         if x['order'] == order
         and x['num_elements'] == num_elem
         and x['mg_level'] != 1
         and x['preconditioner'] == 'Multigrid']
   elif order == 8:
      multigrid_results = [x for x in result
         if x['order'] == order
         and x['num_elements'] == num_elem
         and (x['mg_level'] == 0 or x['mg_level'] == 3)
         and x['preconditioner'] == 'Multigrid']
   else:
      multigrid_results = [x for x in result
         if x['order'] == order
         and x['num_elements'] == num_elem
         and x['preconditioner'] == 'Multigrid']

   multigrid_results.sort(key=lambda k:k['num_pre_smoothe'])
   multigrid_results.sort(key=lambda k:k['mg_level'])

   mg_linestyles = [None, '--', '-.', ':', None, '--', '-.', ':']
   mg_colors = ['blue', 'green', 'purple', 'teal', 'blue', 'green', 'purple']

   fig, ax_res = plt.subplots(1, 1)
   if len(no_precond) > 0:
      ax_res.plot(no_precond[0][:MAX_IT], color='black', label='FGMRES No precond')

   for i, data in enumerate(mg_solver):
      # ax_res.plot(data['residuals'][:MAX_IT], color='teal', marker='.', markersize=6, markevery=4, linestyle=mg_linestyles[i], label=f'MG solv ({data["mg_level"]} lvl, {data["num_pre_smoothe"]} sm)')
      color = 'pink' if data['mg_level'] == 0 else 'indigo'
      ax_res.plot(data['residuals'][:MAX_IT], color=color, linestyle=mg_linestyles[i], label=f'MG solv ({data["mg_level"]} lvl, {data["num_pre_smoothe"]} sm)')
   for i, data in enumerate(fv_ref):
      ax_res.plot(data['residuals'][:MAX_IT], color='orange', linestyle=mg_linestyles[i], label=f'Ref precond')
   for i, data in enumerate(fv_fast):
      ax_res.plot(data['residuals'][:MAX_IT], color='magenta', linestyle=mg_linestyles[i], label=f'Simple FV precond')

   for i, data in enumerate(multigrid_results):
      res = data['residuals']
      ax_res.plot(res[:MAX_IT],
                  color=mg_colors[data['mg_level']],
                  linestyle=mg_linestyles[i],
                  label=f'MG prec ({data["mg_level"]} lvl, {data["num_pre_smoothe"]} sm)')

   ax_res.set_yscale('log')
   ax_res.grid(True)
   ax_res.set_xlabel('Iteration #')
   ax_res.set_ylabel('Residual')

   fig.suptitle(f'Residual evolution, {order}x{num_elem} elem')
   fig.legend()
   fig.savefig(f'{filename}_{order}x{num_elem}.pdf')

def main(args):
   results = read_results(args.results_file)

   if args.plot_iter:
      plot_base_results(results, 'base_fv_precond.png')
      plot_fv(results, 'fv_precond.png')

   if args.plot_residual:
      plot_residual(results, 2, 30, 'residual')
      plot_residual(results, 2, 60, 'residual')
      plot_residual(results, 2, 120, 'residual')

   if args.plot_residual2:
      plot_residual_2(results, 2, 30, 'residual')
      plot_residual_2(results, 2, 60, 'residual')
      plot_residual_2(results, 2, 120, 'residual')
      plot_residual_2(results, 4, 30, 'residual')
      plot_residual_2(results, 4, 60, 'residual')
      plot_residual_2(results, 8, 30, 'residual')

   # for i, order in enumerate([2, 4, 8]):
      # sizes, it, times = get_plot_data(results, order, 'Finite volume', tol=1e-1)
      # ax_it.plot(sizes, it, '*-')
      # ax_time.plot(sizes, times, '*-')


if __name__ == '__main__':
   import argparse

   parser = argparse.ArgumentParser(description='Plot results from automated preconditioner tests')
   parser.add_argument('results_file', type=str, help='File that contains test results')
   parser.add_argument('--plot-iter', action='store_true', help='Plot the iterations and time with respect to various parameters')
   parser.add_argument('--plot-residual', action='store_true', help='Plot residual evolution')
   parser.add_argument('--plot-residual2', action='store_true', help='Plot residual evolution')

   args = parser.parse_args()

   main(args)
