#!/usr/bin/env python3

import os
import sys
import argparse

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "wx_factory")
sys.path.append(main_project_dir)

import compiler.compile_kernels


def main():
    parser = argparse.ArgumentParser(description="Compile the kernels for WxFactory")
    parser.add_argument("backend", choices=["cpp", "cuda"], nargs="?", default="cpp")
    parser.add_argument("--no-clean", action="store_true", help="Skip the clean step")
    args = parser.parse_args()

    modules = ["pde", "operators"]

    for module in modules:
        compiler.compile_kernels.compile(module, args.backend, force=not args.no_clean)


if __name__ == "__main__":
    main()
