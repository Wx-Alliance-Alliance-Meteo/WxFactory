#!/usr/bin/env python3

import numpy

def main(args):
   try:
      states = [numpy.load(f) for f in args.files]
   except:
      print(f'Unable to load given states: {args.files}')
      raise

   differences = [[numpy.linalg.norm(states[i] - states[j]) / numpy.linalg.norm(states[i])
                   for j in range(i+1, len(states))]
                  for i in range(len(states) - 1)]
   diff_s = ' ' + '  '.join([f'{i:9d}' for i in range(len(states))]) + '\n' \
            + '\n'.join([f'{i:3d} '
                         + '  '.join(['         ' for _ in range(i+1)])
                         + ', '.join([f'{d:9.2e}' for d in vector]) for i, vector in enumerate(differences)])
   print(f'Differences: {differences}')
   print(f'Differences: \n{diff_s}')

if __name__ == '__main__':
   import argparse
   parser = argparse.ArgumentParser(description='Compare vector states produced by GEF')
   parser.add_argument('files', type=str, nargs='+', help='Files that contains vector states to compare')
   args = parser.parse_args()
   main(args)
