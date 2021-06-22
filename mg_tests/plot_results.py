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

         def get_precond(num):
            if   num == 0: return 'None'
            elif num == 1: return 'Finite volume'
            elif num == 2: return 'Multigrid'
            else: return 'Big ERROR'

         try:
            data = {
               'order': int(items[0]),
               'num_elements': int(items[1]),
               'preconditioner': get_precond(int(items[2])),
               'precond_tolerance': float(items[3]),
               'mg_level': int(items[4]),
               'mg_smoothe_only': True if int(items[5]) == 1 else False,
               'mg_dt': float(items[6]),
               'num_pre_smoothe': int(items[7]),
               'num_post_smoothe': int(items[8]),
               'cfl': float(items[9]),
               'num_fgmres_it': int(items[11]),
               'fgmres_time': float(items[12]),
               'num_precond_it': int(items[13]),
               'precond_time': float(items[14]),
               'residuals': [float(x) for x in items[16:]]
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

def get_mg_data(results, order, num_elem, smoothe_only, num_smoothes):

   res = [x for x in results
      if x['order'] == order
      and x['num_elements'] == num_elem
      and x['preconditioner'] == 'Multigrid'
      and x['mg_smoothe_only'] == smoothe_only
      and x['num_pre_smoothe'] == num_smoothes]

   # print(f'Res:\n{res}')

   all_data = [(x['mg_dt'], x['num_fgmres_it'], x['fgmres_time']) for x in res]
   all_data.sort()
   dts = [x[0] for x in all_data]
   it = [x[1] for x in all_data]
   times = [x[2] for x in all_data]

   return dts, it, times

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

def plot_fv_mg_solve(results, order, num_elem, filename):
   fig, (ax_it, ax_time) = plt.subplots(2, 1)

   # for i, order in enumerate([2, 4, 8]):
   #    sizes, it, times = get_plot_data(results, order, 'Finite volume', tol=1e-1)
   #    ax_it.plot(sizes, it, 'o-', color=colors[i], label=f'FV ref, o{order}')
   #    ax_time.plot(sizes, times, 'o-', color=colors[i])

   dts, it, times = get_mg_data(results, order, num_elem, True, 1)
   ax_it.plot(dts, it, 'o-', label=f'no solve, 1 smoothe')
   ax_time.plot(dts, times, 'o-')

   dts, it, times = get_mg_data(results, order, num_elem, True, 2)
   ax_it.plot(dts, it, 'o-', label=f'no solve, 2 smoothes')
   ax_time.plot(dts, times, 'o-')

   dts, it, times = get_mg_data(results, order, num_elem, False, 1)
   ax_it.plot(dts, it, 'o-', label=f'w/ solve, 1 smoothe')
   ax_time.plot(dts, times, 'o-')

   dts, it, times = get_mg_data(results, order, num_elem, False, 2)
   ax_it.plot(dts, it, 'o-', label=f'w/ solve, 2 smoothes')
   ax_time.plot(dts, times, 'o-')

   res = [x for x in results
    if x['order'] == order
    and x['num_elements'] == num_elem and x['preconditioner'] == 'Finite volume'
    and x['precond_tolerance'] == 1e-1]
   ax_it.plot(dts, [res[0]['num_fgmres_it'] for i in range(len(dts))], 'o-', label=f'FV precond (fast)')
   ax_time.plot(dts, [res[0]['fgmres_time'] for i in range(len(dts))], 'o-')

   res = [x for x in results
    if x['order'] == order
    and x['num_elements'] == num_elem and x['preconditioner'] == 'None']
   ax_it.plot(dts, [res[0]['num_fgmres_it'] for i in range(len(dts))], 'o-', label=f'No precond')
   ax_time.plot(dts, [res[0]['fgmres_time'] for i in range(len(dts))], 'o-')

   ax_it.set_ylabel('# Iterations')
   ax_time.set_xlabel('pseudo dt')
   ax_time.set_ylabel('Time (s)')

   # ax_it.set_xscale('log')
   # ax_it.set_yscale('log')
   # ax_time.set_xscale('log')
   # ax_time.set_yscale('log')

   fig.suptitle(f'MG with solve vs MG smoothe only, {order}x{num_elem}')
   fig.legend(loc='center right')
   fig.tight_layout()
   fig.savefig(f'{filename}_{order}x{num_elem}.png')

def plot_residual(result, order, num_elem, filename):
   no_precond = [x['residuals'] for x in result
      if x['order'] == order
      and x['num_elements'] == num_elem
      and x['preconditioner'] == 'None'][0][:min(order*num_elem, 120)]
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
                  color=mg_colors[data['mg_dt']],
                  linestyle=mg_linestyles[data['num_pre_smoothe']],
                  label=f'MG dt={data["mg_dt"]}, sm={data["num_pre_smoothe"]}')

   ax_res.set_yscale('log')
   ax_res.grid(True)

   fig.legend()
   fig.savefig(f'{filename}_{order}x{num_elem}.png')

def main(args):
   results = read_results(args.results_file)

   if args.plot_iter:
      plot_base_results(results, 'base_fv_precond.png')
      plot_fv(results, 'fv_precond.png')
      plot_fv_mg_solve(results, 2, 120, 'mg_precond')
      plot_fv_mg_solve(results, 2, 30, 'mg_precond')
      plot_fv_mg_solve(results, 2, 60, 'mg_precond')

   if args.plot_residual:
      plot_residual(results, 2, 30, 'residual')
      plot_residual(results, 2, 60, 'residual')
      plot_residual(results, 2, 120, 'residual')


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

   args = parser.parse_args()

   main(args)
