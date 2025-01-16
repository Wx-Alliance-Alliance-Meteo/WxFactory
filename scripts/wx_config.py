#!/usr/bin/env python3

import argparse
import os
import sys


def main():
    root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    src_dir = os.path.join(root_dir, "wx_factory")
    sys.path.append(root_dir)
    sys.path.append(src_dir)

    from common import load_default_schema

    parser = argparse.ArgumentParser(description="View WxFactory config options")
    parser.add_argument("--list", action="store_true", help="List options")
    parser.add_argument("--list-md", action="store_true", help="List options as a markdown table (for documentation)")
    parser.add_argument("--list-hints", action="store_true", help="List options with their type only")

    args = parser.parse_args()

    schema = load_default_schema()

    if args.list:
        print(schema)
    elif args.list_md:
        print(schema.to_string(markdown=True))
    elif args.list_hints:
        print(schema.type_hints())


if __name__ == "__main__":
    main()
