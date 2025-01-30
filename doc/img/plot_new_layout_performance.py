#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

euler3d_sizes = np.array([10**2 * 8 * 3**3, 10**2 * 8 * 4**3, 12**2 * 8 * 5**3])
sw_sizes = np.array(([25**2 * 5**2, 50**2 * 5**2, 100**2 * 5**2, 150**2 * 5**2]))
euler2d_sizes = np.array([9 * 15 * 5, 19 * 30 * 5, 39 * 60 * 5])

euler3d_old_cpu = np.array([[8.7, 8.7], [43, 35], [173, 172]])
euler3d_old_gpu = np.array([[3.0, 2.3], [5.5, 4.0], [8.3, 7.6]])
euler3d_new_cpu = np.array([[9.5, 9.5], [42, 35], [173, 172]])
euler3d_new_cpu2 = np.array([[9.8, 9.8], [49, 41], [173, 172]])
euler3d_new_gpu = np.array([[3.8, 3.6], [7.4, 6.3], [11.5, 11.2]])

sw_old_cpu = np.array([[0.82, 1.6], [5.8, 8.5], [59, 59], [196, 195]])
sw_new_cpu = np.array([[0.57, 1.1], [4.7, 7.0], [57, 58], [192, 193]])

euler2d_old_cpu = np.array([[3.3, 3.0], [20.1, 20.0], [170, 168]])
euler2d_new_cpu = np.array([[2.4, 2.2], [16.8, 16.1], [155, 151]])

neutral = np.array([1.0, 1.0, 1.0, 1.0])


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), layout="constrained")

ratio3d = np.mean(euler3d_new_cpu, axis=1) / np.mean(euler3d_new_cpu2, axis=1)

ax1.plot(euler2d_sizes, np.mean(euler2d_old_cpu, axis=1) / np.mean(euler2d_new_cpu, axis=1), label="Euler 2D")
ax1.plot(euler2d_sizes, neutral[:3], color="black", linestyle=":")
ax2.plot(sw_sizes, np.mean(sw_old_cpu, axis=1) / np.mean(sw_new_cpu, axis=1), label="Shallow water")
ax2.plot(euler3d_sizes, np.mean(euler3d_old_cpu, axis=1) / np.mean(euler3d_new_cpu, axis=1), label="Euler 3D (CPU)")
ax2.plot(
    euler3d_sizes,
    np.mean(euler3d_old_gpu, axis=1) / (np.mean(euler3d_new_gpu, axis=1) * ratio3d),
    label="Euler 3D (GPU)",
    linestyle="--",
)
ax2.plot(sw_sizes, neutral, color="black", linestyle=":")
for ax in ax1, ax2:
    ax.set_xscale("log")
    ax.legend()
    ax.set_xlabel("Number of grid points")
    ax.set_ylabel("Speedup")

# fig.suptitle(f"Performance improvement of new vs old layout")


plt.savefig(f"layout_perf.eps")
plt.close(fig)
