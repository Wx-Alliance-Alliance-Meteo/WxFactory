#!/usr/bin/env python3

import sys

from graphx import plot_field_from_file

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('Need 2 arguments')
        exit(1)

    geom_name = sys.argv[1]
    field_name = sys.argv[2]
    plot_field_from_file(geom_name, field_name)

