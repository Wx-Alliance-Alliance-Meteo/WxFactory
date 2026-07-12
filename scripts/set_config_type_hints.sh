#!/usr/bin/env bash
#
# Regenerate the static type-hint block in wx_factory/common/configuration.py from the
# configuration schema. Thin wrapper around the wx_factory.common.config_hints module,
# which does the generation (and provides a --check mode used by the tests and CI).

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WX_DIR=$(dirname "${SCRIPT_DIR}")

cd "${WX_DIR}" && exec python3 -m wx_factory.common.config_hints --write
