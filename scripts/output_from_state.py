#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wx_factory.common.graphx import image_field
from wx_factory.output import load_state
from wx_factory.simulation import Simulation


def vertical_wind(filename, simulations):
    state, config = load_state(filename, device=torch.device("cpu"))
    if config.grid_type != "cartesian3d" or config.equations != "euler":
        raise ValueError("State plotting supports Cartesian Euler slabs only")
    simulation = simulations.get(config.config_content)
    if simulation is None:
        config.pytorch_device = "cpu"
        config.output_freq = config.stat_freq = config.save_state_freq = 0
        simulation = Simulation(config, quiet=True)
        simulations[config.config_content] = simulation
    geometry = simulation.geometry
    if state.shape != simulation.Q.shape:
        raise ValueError(f"Saved state shape {state.shape} does not match the configured grid {simulation.Q.shape}")
    wind = simulation.initial_state.metric.physical_winds(state[1:4])[2]
    block = geometry.to_single_block(wind)
    wind_xz = block[:, block.shape[1] // 2, :]
    return geometry, config.case_number, wind_xz


def main(args):
    simulations = {}
    minimum, maximum = float("inf"), float("-inf")
    for filename in args.input_states:
        _, _, wind = vertical_wind(filename, simulations)
        minimum = min(minimum, float(wind.min()))
        maximum = max(maximum, float(wind.max()))

    if minimum == maximum:
        minimum -= 1
        maximum += 1

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    for index, filename in enumerate(args.input_states):
        geometry, case, wind = vertical_wind(filename, simulations)
        output = f"{args.output_file}_{case}_w_{index:08d}"
        image_field(geometry, wind, output, minimum, maximum, 100, "w (m/s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot physical vertical wind from Cartesian Euler restart states.")
    parser.add_argument("input_states", nargs="+", help="Files containing state vectors")
    parser.add_argument("-o", "--output-file", required=True, help="Output filename prefix")
    main(parser.parse_args())
