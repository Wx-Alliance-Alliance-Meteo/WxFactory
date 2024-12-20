#!/usr/bin/env python3

import os
import sys
import argparse

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "wx_factory")
sys.path.append(main_project_dir)

import compiler.compile_utils


def main():
    parser = argparse.ArgumentParser(description="Compile the kernels for WxFactory")
    parser.add_argument("choice", choices=["cpp", "cuda"], default="cpp")
    args = parser.parse_args()

    compiler.compile_utils.compile(args.choice, True)


if __name__ == "__main__":
    main()
