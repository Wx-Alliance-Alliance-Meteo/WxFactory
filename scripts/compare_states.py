#!/usr/bin/env python3

import os
import sys

import numpy
import torch
from torch import Tensor

# We assume the script is in a subfolder of the main project
main_wx_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(main_wx_dir)

from wx_factory.output import load_state
from wx_factory.common import Configuration


def rel_diff(a: Tensor, b: Tensor):
    num_vars = a.shape[0]
    a_norms = a.reshape(num_vars, -1).norm(dim=1)
    a_norms = torch.where(a_norms == 0.0, 1.0, a_norms)
    diffs = (b - a).reshape(num_vars, -1).norm(dim=1) / a_norms
    return diffs.norm().item()


def pair_diff(a: tuple[Tensor, Configuration], b: tuple[Tensor, Configuration]):

    a_state, a_config = a
    b_state, b_config = b

    # Check that the grids are the same
    if not (
        a_config.grid_type == b_config.grid_type
        and a_config.num_solpts == b_config.num_solpts
        and a_config.num_elements_horizontal == b_config.num_elements_horizontal
        and a_config.num_elements_vertical == b_config.num_elements_vertical
    ):
        raise ValueError("Grids are not the same!")

    s0, s1 = (
        (a_state, b_state) if a_config.grid_type == "cubed_sphere" else (a_state.swapaxes(0, 1), b_state.swapaxes(0, 1))
    )

    return rel_diff(s0, s1)


def main(args):
    try:
        states = [load_state(f) for f in args.files]
    except:
        print(f"Unable to load given states: {args.files}")
        raise

    # print(f"shape = {states[0].shape}")
    norms = torch.tensor([s.norm().item() for s, _ in states])
    diff_norms = torch.tensor([pair_diff(states[0], s) for s in states[1:]])

    torch.set_printoptions(precision=3)

    print(f"Norms: {norms}")
    print(f"Diff norms: {diff_norms}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare vector states produced by WxFactory")
    parser.add_argument("files", type=str, nargs="+", help="Files that contains vector states to compare")
    args = parser.parse_args()
    main(args)
