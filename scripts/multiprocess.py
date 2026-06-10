import multiprocessing as mp
import subprocess


def run_job(config_file):
    cmd = ["mpirun", "-n", "6", "./WxFactory", config_file]
    subprocess.run(cmd)


if __name__ == "__main__":

    configs = [
        "config/ERA5_CS_202001.ini",
        "config/ERA5_CS_202002.ini",
        "config/ERA5_CS_202003.ini",
        "config/ERA5_CS_202004.ini",
    ]

    with mp.Pool(4) as pool:
        pool.map(run_job, configs)
