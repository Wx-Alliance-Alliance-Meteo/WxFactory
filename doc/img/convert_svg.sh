#!/usr/bin/env bash

if [ ! $(which inkscape) ]; then
    echo "Need to have inkscape available!"
    exit -1
fi

for file in *.svg; do
    base=$(basename $file .svg)
    inkscape ${base}.svg -o ${base}.eps --export-ignore-filters  --export-ps-level=3
done

