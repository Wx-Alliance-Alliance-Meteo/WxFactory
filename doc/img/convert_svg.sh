#!/usr/bin/env bash

set -e

if ! command -v rsvg-convert >/dev/null; then
    echo "Install rsvg-convert to build the presentation figures."
    exit 1
fi

for file in "$(dirname "$0")"/*.svg; do
    rsvg-convert --format=pdf --output="${file%.svg}.pdf" "$file"
done
