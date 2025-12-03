#!/usr/bin/env python3

import os
import sys
import argparse

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "wx_factory")
sys.path.append(main_project_dir)

import compiler.compile_kernels


def main():
    parser = argparse.ArgumentParser(description="Compile the kernels for WxFactory")
    parser.add_argument("backend", choices=["cpp", "cuda", "omp"], nargs="?", default="cpp")
    parser.add_argument("--no-clean", action="store_true", help="Skip the clean step")
    parser.add_argument("--modules", type=str, nargs="+", default=["pde", "operators", "exchanges"])
    parser.add_argument("--proc-name", type=str, default="")
    args = parser.parse_args()

    if args.proc_name != "":
        compiler.compile_kernels._proc_name = args.proc_name

    for module in args.modules:
        compiler.compile_kernels.compile(module, args.backend, force=not args.no_clean)


if __name__ == "__main__":
    main()
