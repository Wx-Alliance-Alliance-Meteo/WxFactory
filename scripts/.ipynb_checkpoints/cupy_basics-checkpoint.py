#!/usr/bin/env python3

import numpy as np
import cupy as cp

def dev_info(id):
   with cp.cuda.Device(id) as dev:
      free_mem, total_mem = dev.mem_info
      kb = 1024
      mb = kb * kb
      gb = kb * kb * kb
      print(f'Device {id}: {free_mem / gb :.1f}/{total_mem / gb :.1f} GB available')

def main():
   x_gpu = cp.array([1, 2, 3])
   norm_gpu = cp.linalg.norm(x_gpu)
   print(f'norm_gpu = {norm_gpu}')

   x = np.array([1, 2, 3])
   norm = np.linalg.norm(x)
   print(f'norm =     {norm}')

   print(f'device = {x_gpu.device}')

   cp.show_config()

   num_devices = cp.cuda.runtime.getDeviceCount()
   print(f'There are {num_devices} devices')

   for i in range(num_devices):
      try:
         dev_info(i)
      except:
         print(f'{i} is a wrong number')



if __name__ == '__main__':
   main()
