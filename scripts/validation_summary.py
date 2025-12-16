#!/usr/bin/env python3

import os
import sys
import argparse
from types import SimpleNamespace

from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen
from io import BytesIO
import numpy as np
import requests
import validation

import compare_outputs
import generate_potential_temperature_plot
import generate_hovmoller_diagram

ALLOWED_TYPES = [".npy", ".nc"]

def _get_reference(reference_url: str, store_reference_path: str):
    """
    Description: Download and stores a reference sv or netcdf
    """
    
    if os.path.isfile(store_reference_path):
        print(f"File already exists: {store_reference_path}")
        return store_reference_path
    
    # Download
    r = requests.get(reference_url)
    print(r.status_code, r.headers.get("Content-Type"), len(r.content))
    r.raise_for_status()
    if not r.content:
        raise RuntimeError("Empty response body")

    # Save
    os.makedirs(os.path.dirname(store_reference_path), exist_ok=True)
    with open(store_reference_path, "wb") as f:
        f.write(r.content)
    print(f"Saved file to: {store_reference_path}")
        
    return store_reference_path

def _get_compare_summary(target_path: str, reference):
    
    extension = os.path.splitext(target_path)[1]
    extension_reference = os.path.splitext(reference)[1]
    
    if extension not in ALLOWED_TYPES:
        raise ValueError(f"Unsupported type: {extension}")
    if extension != extension_reference:
        raise ValueError(f"No reference type matches type {extension}")
    
    if target_path.endswith(".nc"):
        report = compare_outputs.run(
            target_path,
            reference,
            input_type="netcdf",
            vars=["P", "rho", "theta"],
        )
    elif target_path.endswith(".npy"):
        report = compare_outputs.run(
            target_path,
            reference,
            input_type="sv",
            vars=["P", "rho", "theta"],
        )
        
    return report

def _plot_potential(source_path: str, output_file: str):
    generate_potential_temperature_plot.run(data_file=source_path, output_file=output_file)
    print(f"Potential temperature plot saved at {output_file}")
    return output_file

def _plot_hovmoller(source_path: str, output_file: str):
    generate_hovmoller_diagram.run(data_file=source_path, output_file=output_file)
    print(f"Hovmoller diagram saved at {output_file}")
    return output_file

def _get_image(source_url: str, store_path: str):
    
    if os.path.exists(store_path):
        print(f"File already exists: {store_path}")
        return store_path

    os.makedirs(os.path.dirname(store_path), exist_ok=True)

    response = requests.get(source_url, stream=True)
    response.raise_for_status()

    with open(store_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return store_path

def main(args):
    
    # Aggregate different reports
    reference_path = _get_reference(args.reference_url, args.store_reference_path)
    compare_report_list = _get_compare_summary(args.target_file, reference_path)
    
    # Plots (only compatible with netcdf)
    isPlot = False
    if (args.target_file.endswith(".nc")):
        isPlot = True
    
        potential_plot = _plot_potential(args.target_file, args.target_potential_plot_store_path)
        reference_potential_plot = _get_image(args.reference_potential_plot_url, args.store_reference_path_potential)
        
        hovmoller_plot = _plot_hovmoller(args.target_file, args.target_hovmoller_plot_store_path)
        reference_hovmoller_plot = _get_image(args.reference_hovmoller_plot_url, args.store_reference_path_hovmoller)    
    
    else:
        print("File type not supported for temperature potential and hovmoller plots")
    
    # Build summary
    report = validation.Report(title="Validation Report")
    
    summary = validation.Section(title="Summary")
    for compare_report in compare_report_list:
        summary.blocks.append(validation.TextBlock(compare_report))
    report.add(summary)
    
    if isPlot:
        potential = validation.Section(title="Potential")
        potential.blocks.append(validation.ImageBlock(title="Potential Plot", image_path=potential_plot, caption="Potential plot"))
        potential.blocks.append(validation.ImageBlock(title="Reference Potential Plot", image_path=reference_potential_plot))
        report.add(potential)
        
        hovmoller = validation.Section(title="Hovmoller")
        hovmoller.blocks.append(validation.ImageBlock(title="Hovmoller Diagram", image_path=hovmoller_plot, caption="Hovmoller diagram"))
        hovmoller.blocks.append(validation.ImageBlock(title="Reference Hovmoller Diagram", image_path=reference_hovmoller_plot))
        report.add(hovmoller)
        
    report.save("scripts/validation/report.html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Generate a [html, ] report of comparison between two [netcdf, state vectors] grid.
        We gather the nrmse, l2 and spectral error across the different panels.
        If the grids are not the same size, we try [linear, cubic, quintic] interpolation with the finer grid as reference.
    """
    )
    # Target file of evaluation
    parser.add_argument("target_file", help="")
    parser.add_argument("--target_potential_plot_store_path", default = "/home/ngv000/repos/WxFactory/scripts/validation/tmp/target_potential.png", type=str)
    parser.add_argument("--target_hovmoller_plot_store_path", default = "/home/ngv000/repos/WxFactory/scripts/validation/tmp/target_hovmoller.png", type=str)

    # Reference / baseline file
    parser.add_argument("--reference_url", default="https://web.science.gc.ca/~ngv000/WxFactory/reference_state.npy", type=str)
    # parser.add_argument("--reference_url", default="https://web.science.gc.ca/~ngv000/WxFactory/toy_netcdf.nc", type=str)
    # parser.add_argument("--store_reference_path", default="/home/ngv000/repos/WxFactory/scripts/validation/tmp/reference.nc", type=str)
    parser.add_argument("--store_reference_path", default="/home/ngv000/repos/WxFactory/scripts/validation/tmp/reference.npy", type=str)
    parser.add_argument("--store_reference_path_potential", default="/home/ngv000/repos/WxFactory/scripts/validation/tmp/potential_ref_plot.jpg", type=str)
    parser.add_argument("--store_reference_path_hovmoller", default="/home/ngv000/repos/WxFactory/scripts/validation/tmp/hovmoller_ref_plot.jpg", type=str)

    parser.add_argument("--reference_potential_plot_url", default = "https://web.science.gc.ca/~ngv000/WxFactory/rubber_duck.jpg", type=str)
    parser.add_argument("--reference_hovmoller_plot_url", default = "https://web.science.gc.ca/~ngv000/WxFactory/mr_potato.jpg", type=str)

    args = parser.parse_args()
    main(args)