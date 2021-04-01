#!/usr/bin/env python3

import sys

import scipy
import scipy.sparse.linalg
from scipy.sparse import load_npz


def main(file1, file2):
    j1 = load_npz(file1)
    j2 = load_npz(file2)

    diff_mat = j2 - j1

    total_diff = scipy.sparse.linalg.norm(diff_mat)
    print(f'Total diff: {total_diff}')


if __name__ == '__main__':

    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])

