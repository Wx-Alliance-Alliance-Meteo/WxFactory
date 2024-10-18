#!/usr/bin/env python3

import os
import sys

import numpy

# We assume the script is in a subfolder of the main project
main_gef_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(main_gef_dir)

from common.definitions      import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta
from common.graphx           import image_field
from geometry                import Cartesian2D
from main_gef                import create_geometry
from output.output_cartesian import output_step
from output.output_manager   import OutputManager
from output.state            import load_state

def main(args):

   min_rho   = 1.0e100
   min_u     = min_rho
   min_w     = min_rho
   min_theta = min_rho
   max_rho   = -min_rho
   max_u     = -min_u
   max_w     = -min_w
   max_theta = -min_theta
   for s in args.input_states:
      state, param = load_state(s)
      min_rho = min(min_rho, state[idx_2d_rho].min())
      max_rho = max(max_rho, state[idx_2d_rho].max())
      min_u   = min(min_u, (state[idx_2d_rho_u] / state[idx_2d_rho]).min())
      max_u   = max(max_u, (state[idx_2d_rho_u] / state[idx_2d_rho]).max())
      min_w   = min(min_w, (state[idx_2d_rho_w] / state[idx_2d_rho]).min())
      max_w   = max(max_w, (state[idx_2d_rho_w] / state[idx_2d_rho]).max())
      min_theta = min(min_theta, (state[idx_2d_rho_theta] / state[idx_2d_rho]).min())
      max_theta = max(max_theta, (state[idx_2d_rho_theta] / state[idx_2d_rho]).max())

   file_name = lambda var, id: f'{args.output_file}_{param.case_number}_{var}_{i:08d}'

   for i, s in enumerate(args.input_states):
      state, param = load_state(s)
      geom = create_geometry(param, None)
      if isinstance(geom, Cartesian2D):
         # image_field(geom, state[idx_2d_rho], file_name('rho', i), min_rho, max_rho, 25, 'rho (units=?)')
         # image_field(geom, state[idx_2d_rho_u] / state[idx_2d_rho], file_name('u', i), min_u, max_u, 25, 'u (m/s)')
         image_field(geom, state[idx_2d_rho_w] / state[idx_2d_rho], file_name('w', i), min_w, max_w, 100, 'w (m/s)')
         # image_field(geom, state[idx_2d_rho_theta] / state[idx_2d_rho], file_name('theta', i), min_theta, max_theta, \
         #             25, 'theta (units=?)')
         output_step(state, geom, param, file_name("", i))

if __name__ == '__main__':
   import argparse

   parser = argparse.ArgumentParser(description='Generate output file(s) from simulation states. ONLY 2D FOR NOW.')
   parser.add_argument('input_states', nargs='+', type=str, help='File(s) containing state vectors')
   parser.add_argument('-o', '--output-file', required=True, type=str, help='Name of the output file')

   parsed_args = parser.parse_args()
   main(parsed_args)
