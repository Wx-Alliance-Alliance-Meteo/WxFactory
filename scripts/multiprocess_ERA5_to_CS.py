import multiprocessing as mp
import subprocess
import argparse


def run_job(config_file):
    cmd = ["mpirun", "-n", "6", "./WxFactory", config_file]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run WxFactory jobs in parallel with multiple config files")

    # Accept one or more config files
    parser.add_argument("configs", nargs="+", help="List of config files to process")

    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")

    args = parser.parse_args()

    with mp.Pool(args.workers) as pool:
        pool.map(run_job, args.configs)


if __name__ == "__main__":
    main()
