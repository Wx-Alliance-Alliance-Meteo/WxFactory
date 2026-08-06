"""Generate and verify the static type-hint block in :mod:`wx_factory.common.configuration`.

``Configuration`` populates its attributes dynamically from the JSON schema, so nothing in the
class body tells an IDE (or a type checker) that ``config.num_elements_horizontal`` is an ``int``.
To recover autocompletion and static checking, the class carries a block of ``name: type``
annotations generated from the schema, delimited by the ``START``/``END`` markers below.

This module is the single tool that writes that block and checks it is up to date:

    python -m wx_factory.common.config_hints --write    # regenerate the block in place
    python -m wx_factory.common.config_hints --check     # exit non-zero if the block is stale

The ``--check`` mode is what CI and the test suite run, so that editing the schema without
regenerating the annotations fails the build instead of silently letting the types drift.
"""

import argparse
import difflib
import sys
from pathlib import Path

from .configuration_schema import ConfigurationSchema

# Repo root is three levels up: wx_factory/common/config_hints.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = _REPO_ROOT / "config" / "config-format.json"
CONFIG_PATH = _REPO_ROOT / "wx_factory" / "common" / "configuration.py"

START_MARKER = "    # --- START type hints --- automatically generated (do not touch)"
END_MARKER = "    # --- END type hints --- automatically generated (do not touch)"

_INDENT = "    "


def load_schema(schema_path: Path = SCHEMA_PATH) -> ConfigurationSchema:
    """Load the configuration schema from its JSON description."""
    return ConfigurationSchema(schema_path.read_text())


def build_hint_lines(schema: ConfigurationSchema) -> list[str]:
    """Return the annotation lines (``    name: type``), sorted by option name.

    ``CaseSensitiveStr`` reports its type as ``cs-str``, which is not a valid Python type name;
    it is mapped to ``str`` here (the runtime value really is a ``str`` subclass).

    A field may appear more than once in ``schema.fields`` (dependency-gated fields can be listed
    repeatedly), so lines are de-duplicated before sorting."""
    lines = set()
    for field in schema.fields:
        type_name = field.typename().replace("cs-str", "str")
        lines.add(f"{_INDENT}{field.name}: {type_name}")
    return sorted(lines)


def render_block(schema: ConfigurationSchema) -> list[str]:
    """Return the full marker-delimited block, including the marker lines."""
    return [START_MARKER, *build_hint_lines(schema), END_MARKER]


def splice_block(source: str, block_lines: list[str]) -> str:
    """Replace the marker-delimited region of ``source`` with ``block_lines``.

    Everything strictly between the START and END markers is replaced; the markers themselves
    are rewritten from ``block_lines`` so they stay canonical."""
    lines = source.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == START_MARKER.strip())
        end = next(i for i, l in enumerate(lines) if l.strip() == END_MARKER.strip())
    except StopIteration as e:
        raise ValueError(
            f"Could not find the type-hint markers in {CONFIG_PATH}. Expected lines:\n{START_MARKER}\n...\n{END_MARKER}"
        ) from e
    if end < start:
        raise ValueError(f"END marker appears before START marker in {CONFIG_PATH}")

    new_lines = lines[:start] + block_lines + lines[end + 1 :]
    result = "\n".join(new_lines)
    if source.endswith("\n"):
        result += "\n"
    return result


def write(schema: ConfigurationSchema, config_path: Path = CONFIG_PATH) -> bool:
    """Regenerate the block in ``config_path``. Return ``True`` if the file changed."""
    source = config_path.read_text()
    updated = splice_block(source, render_block(schema))
    if updated == source:
        return False
    config_path.write_text(updated)
    return True


def check(schema: ConfigurationSchema, config_path: Path = CONFIG_PATH) -> list[str]:
    """Return a unified diff of what would change if the block were regenerated.

    An empty list means the committed block is up to date."""
    source = config_path.read_text()
    updated = splice_block(source, render_block(schema))
    if updated == source:
        return []
    return list(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"{config_path} (committed)",
            tofile=f"{config_path} (regenerated)",
        )
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="Verify the committed type-hint block matches the schema; exit 1 if stale (default).",
    )
    group.add_argument(
        "--write", action="store_true", help="Regenerate the type-hint block in configuration.py in place."
    )
    args = parser.parse_args(argv)

    schema = load_schema()

    if args.write:
        changed = write(schema)
        print("Updated type-hint block." if changed else "Type-hint block already up to date.")
        return 0

    # Default action is --check
    diff = check(schema)
    if diff:
        sys.stderr.write("".join(diff))
        sys.stderr.write(
            "\nConfiguration type hints are out of date. "
            "Run `python -m wx_factory.common.config_hints --write` and commit the result.\n"
        )
        return 1

    print("Configuration type hints are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
