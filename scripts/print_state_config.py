#!/usr/bin/env python3

import argparse
import os
import sys

# We assume the script is in a subfolder of the main project
main_wx_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../wx_factory")
sys.path.append(main_wx_dir)

from output import load_state


def main(args):

    _, config = load_state(args.config)
    print(f"{config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the configuration that was stored with a certain state vector")
    parser.add_argument("config", type=str)

    main(parser.parse_args())
