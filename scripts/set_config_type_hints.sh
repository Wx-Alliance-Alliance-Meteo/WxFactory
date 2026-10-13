#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(dirname "$script_dir")"

cd "$project_dir" || exit 1
exec python3 -m wx_factory.common.config_hints --write
