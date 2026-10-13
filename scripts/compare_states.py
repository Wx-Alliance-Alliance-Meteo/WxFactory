#!/usr/bin/env python3

import os
import sys

import torch
from torch import Tensor

main_wx_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(main_wx_dir)

from wx_factory.common import Configuration
from wx_factory.output import load_state


def rel_diff(a: Tensor, b: Tensor):
    num_variables = a.shape[0]
    reference_norms = a.reshape(num_variables, -1).norm(dim=1)
    reference_norms = torch.where(reference_norms == 0.0, 1.0, reference_norms)
    relative_errors = (b - a).reshape(num_variables, -1).norm(dim=1) / reference_norms
    return relative_errors.norm().item()


def pair_diff(a: tuple[Tensor, Configuration], b: tuple[Tensor, Configuration]):

    a_state, a_config = a
    b_state, b_config = b

    if not (
        a_config.grid_type == b_config.grid_type
        and a_config.num_solpts == b_config.num_solpts
        and a_config.num_elements_horizontal == b_config.num_elements_horizontal
        and a_config.num_elements_vertical == b_config.num_elements_vertical
    ):
        raise ValueError("Grids are not the same!")

    s0, s1 = (
        (a_state, b_state)
        if a_config.grid_type in ("cubed_sphere", "cartesian3d")
        else (a_state.swapaxes(0, 1), b_state.swapaxes(0, 1))
    )

    return rel_diff(s0, s1)


def main(args):
    try:
        states = [load_state(f) for f in args.files]
    except (OSError, ValueError):
        print(f"Unable to load given states: {args.files}")
        raise

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
