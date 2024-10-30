#!/usr/bin/env python3

import os
import sys

main_project_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append(main_project_dir)

from compiler.compiler_module import build_libraries


def main():
    build_libraries(True)


if __name__ == "__main__":
    main()
