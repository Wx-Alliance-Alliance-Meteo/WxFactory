#!/usr/bin/env python3

import argparse
import concurrent.futures
import os
import sys
import threading
import time

import numpy as np
import requests
from tqdm import tqdm
from tqdm.contrib import itertools

src_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "wx_factory")
sys.path.append(src_dir)

from output.state import load_state

filenames = [f"state_vector_9bdfe22b301d.00000{i:03d}.npy" for i in range(0, 150, 25)]

progress_lock = threading.Lock()


def _download_file(url, filename, force=False):
    """Download a file from a URL with a progress bar. We don't re-download if the file exists (based on size),
    unless force is True.

    Args:
        url (str): The URL of the file to download.
        filename (str): The local path where the file will be saved.
        force (bool): If True, force re-download even if the file exists.
    """
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024 * 8

    if os.path.isfile(filename) and os.path.getsize(filename) == total_size and not force:
        print(f"{filename} already exists, won't download")
        return

    with open(filename, "wb") as file, tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as progress_bar:
        for data in r.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")


def _get_reference(reference_dir: str, save_dir: str, force_download: bool = False):
    """
    Download and store the reference files locally.
    """
    print(f"Saving reference files to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    for filename in filenames:
        _download_file(os.path.join(reference_dir, filename), os.path.join(save_dir, filename), force=force_download)


def _get_rmse(a, b, progress_bar=None):
    rmse = np.sqrt(np.mean((b - a) ** 2))
    range = np.max(a) - np.min(a)
    nrmse = rmse
    if range > 0.0:
        nrmse /= range

    if progress_bar is not None:
        with progress_lock:
            progress_bar.update(1)

    return rmse, nrmse


def _get_diff(ref, current, progress_bar=None):
    result = []
    for p_ref, p_cur in zip(ref, current):  # Iterate over panels
        result.append([_get_rmse(a, b, progress_bar) for a, b in zip(p_ref, p_cur)])

    return np.array(result)


def _single_compare(filename, results_dir, reference_dir, progress_bar):
    result_file = os.path.join(results_dir, filename)
    reference_file = os.path.join(reference_dir, filename)

    if not os.path.isfile(result_file) or not os.path.isfile(reference_file):
        return FileNotFoundError(f"Missing file(s) for comparison: {result_file}, {reference_file}")

    print(f"\rLoading file {filename} for comparison...")
    if progress_bar.total != 0:
        progress_bar.refresh()

    ref_state = load_state(reference_file)[0]
    cur_state = load_state(result_file)[0]

    if progress_bar.total == 0:
        with progress_lock:
            progress_bar.total = len(filenames) * ref_state.shape[0] * ref_state.shape[1]
            progress_bar.refresh()

    report = _get_diff(ref_state, cur_state, progress_bar)

    return report


def _get_compare_summary(results_dir: str, reference_dir: str, max_workers: int = 2):

    reports = [None for _ in range(len(filenames))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        progress_bar = tqdm(total=0, unit="comparisons", desc="Comparing results to reference", delay=1)
        future_to_index = {}
        for i, filename in enumerate(filenames):
            job = executor.submit(_single_compare, filename, results_dir, reference_dir, progress_bar)
            future_to_index[job] = i
            time.sleep(0.1)  # Stagger the submissions to reduce contention

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            report = future.result()

            if isinstance(report, Exception):
                raise report

            reports[index] = report[..., 1]

    progress_bar.close()
    return np.stack(reports)


def main(args):

    _get_reference(args.reference_url, args.store_ref, args.force_download)
    comparison = _get_compare_summary(args.results_dir, args.store_ref, args.max_concurrent)

    np.set_printoptions(precision=2)
    # print(f"report: \n{comparison}")

    avg_diff = np.average(comparison)
    max_diff = np.max(comparison)

    is_pass = avg_diff < args.pass_threshold_avg
    is_pass_max = max_diff < args.pass_threshold_max

    print(
        f"avg diff {'PASS' if is_pass else 'FAIL'} "
        f"with error {avg_diff:.2e} {'<' if is_pass else '>'} threshold {args.pass_threshold_avg}"
    )
    print(
        f"max diff {'PASS' if is_pass_max else 'FAIL'} "
        f"with error {max_diff:.2e} {'<' if is_pass else '>'} threshold {args.pass_threshold_max}"
    )

    return is_pass and is_pass_max


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
            Compare a given benchmark result to a reference solution.
            We gather the normalized root mean square error of the result to the reference solution for each variable
            individually, one cubed-sphere panel at a time.
        """
    )

    parser.add_argument("results_dir", help="Directory containing the benchmark results to evaluate")
    parser.add_argument(
        "--reference-url",
        default="https://hpfx.collab.science.gc.ca/~sidr000/nfs/WxFactory_results/8th_deg_benchmark/",
        type=str,
        help="URL directory from which to download reference files",
    )
    parser.add_argument("--store-ref", type=str, default="./results/ref", help="Directory to store reference files")
    parser.add_argument(
        "--force-download", type=bool, default=False, help="Whether to force re-download of reference files"
    )
    parser.add_argument(
        "--pass-threshold-avg",
        default=1e-7,
        type=float,
        help="Threshold for passing the benchmark (average of all variables)",
    )
    parser.add_argument(
        "--pass-threshold-max",
        default=1e-6,
        type=float,
        help="Threshold for passing the benchmark (maximum of all variables)",
    )
    parser.add_argument(
        "--max-concurrent", default=1, type=int, help="Maximum number of concurrent comparisons (limited by memory)"
    )

    args = parser.parse_args()
    main(args)
