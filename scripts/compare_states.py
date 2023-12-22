#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy

def main(args):
   try:
      states = [numpy.load(f) for f in args.files]
   except:
      print(f'Unable to load given states: {args.files}')
      raise

   norms = numpy.array([numpy.linalg.norm(s) for s in states])
   diff_arrays = [states[0] - s for s in states]
   diff_norms = numpy.array([numpy.linalg.norm(d) for d in diff_arrays])
   differences = [[numpy.linalg.norm(states[i] - states[j]) / numpy.linalg.norm(states[i])
                   for j in range(i+1, len(states))]
                  for i in range(len(states) - 1)]
   diff_s = ' ' + '  '.join([f'{i:9d}' for i in range(len(states))]) + '\n' \
            + '\n'.join([f'{i:3d} '
                         + '  '.join(['         ' for _ in range(i+1)])
                         + ', '.join([f'{d:9.2e}' for d in vector]) for i, vector in enumerate(differences)])

   numpy.set_printoptions(precision=3)

   print(f'Norms: {norms}')
   print(f'Diff norms: {diff_norms}')
   print(f'Differences (relative): {differences}')
   print(f'Differences (relative): \n{diff_s}')

   if len(diff_arrays[0].shape) == 3:
      fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(2, 2)
      a0 = ax00.imshow(diff_arrays[1][0], interpolation='nearest')
      a1 = ax01.imshow(diff_arrays[1][1], interpolation='nearest')
      a2 = ax10.imshow(diff_arrays[1][2], interpolation='nearest')
      a3 = ax11.imshow(diff_arrays[1][3], interpolation='nearest')
      fig.colorbar(a0, ax=ax00)
      fig.colorbar(a1, ax=ax01)
      fig.colorbar(a2, ax=ax10)
      fig.colorbar(a3, ax=ax11)
      fig.savefig('test.png')

if __name__ == '__main__':
   import argparse
   parser = argparse.ArgumentParser(description='Compare vector states produced by GEF')
   parser.add_argument('files', type=str, nargs='+', help='Files that contains vector states to compare')
   args = parser.parse_args()
   main(args)
