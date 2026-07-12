#!/usr/bin/env python3

import numpy as np
import cupy as cp


def dev_info(id):
    with cp.cuda.Device(id) as dev:
        free_mem, total_mem = dev.mem_info
        kb = 1024
        mb = kb * kb
        gb = kb * kb * kb
        print(f"Device {id}: {free_mem / gb :.1f}/{total_mem / gb :.1f} GB available")
        attr = dev.attributes
        if id == 0:
            try:
                print(
                    f"  Max block size: {attr['MaxBlockDimX']} x {attr['MaxBlockDimY']} x {attr['MaxBlockDimZ']} "
                    f"(max total threads {attr['MaxThreadsPerBlock']})"
                )
                print(f"  Max grid size: {attr['MaxGridDimX']} x {attr['MaxGridDimY']} x {attr['MaxGridDimZ']}")
                print(f"  Max blocks per SM: {attr['MaxBlocksPerMultiprocessor']}")
                print(f"  Max threads per SM: {attr['MaxThreadsPerMultiProcessor']}")
                num_sm = attr["MultiProcessorCount"]
                print(f"  Number of SMs: {num_sm} -> {num_sm * 128} cores (?)")
                print(f"  Max registers per block: {attr['MaxRegistersPerBlock']}")
                print(f"  Max shared mem per block: {attr['MaxSharedMemoryPerBlock'] / kb:.1f} kB")
                print(f"  L2 cache size: {attr['L2CacheSize'] / mb:.1f} MB")
                print(f"  Constant memory: {attr['TotalConstantMemory'] / kb:.1f} kB")
                print(f"  Global memory: {total_mem / gb :.2f} GB")
                print(f"  Direct RDMA supported? {attr['GPUDirectRDMASupported']}")
            except KeyError:
                pass
            # print(f'attributes: \n{dev.attributes}')


def main():
    x_gpu = cp.array([1, 2, 3])
    norm_gpu = cp.linalg.norm(x_gpu)
    print(f"norm_gpu = {norm_gpu}")

    x = np.array([1, 2, 3])
    norm = np.linalg.norm(x)
    print(f"norm =     {norm}")

    print(f"device = {x_gpu.device}")

    cp.show_config()

    num_devices = cp.cuda.runtime.getDeviceCount()
    print(f"There are {num_devices} devices")

    for i in range(num_devices):
        try:
            dev_info(i)
        except:
            print(f"{i} is a wrong number")


if __name__ == "__main__":
    main()
