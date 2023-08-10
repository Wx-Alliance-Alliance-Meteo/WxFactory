#!/usr/bin/env bash

ffmpeg -framerate 15 -i bubble_cliff/bubble_4_00000%3d.png -c:v libx264 -r 30 output.mp4

