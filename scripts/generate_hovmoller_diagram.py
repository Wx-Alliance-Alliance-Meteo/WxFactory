#!/usr/bin/env python3

import sys
import os

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
src_dir = os.path.join(root_dir, "wx_factory")
sys.path.append(root_dir)
sys.path.append(src_dir)

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import output
from mpi4py import MPI

# Load data
data = nc.Dataset('./tests/data/temp/out.nc', 'r')
_, T0 = output.InputManager.read_config_from_save_file('./tests/data/temp/old/state_vector_47170bb8616a.00000000.npy', MPI.COMM_WORLD)

T0_theta = T0[:, 4, ...] / T0[:, 0, ...]

# Setup figure
fig = plt.figure(figsize=(5, 5))

times = data['time'][:]
θ = data['theta']
elev = data['elev']
lons = data['lons'][:]
lats = data['lats'][:]
(ntime, npanels, nk, nj, ni) = θ.shape  # Dimensions: (time, panel, height, lat, lon)

hk = 5  # Choose specific height index (modify as needed)
yrow = nj - 1  # Fixed latitude row

# Initialize lists for longitude, time, and delta theta
all_lons = []
all_times = []
all_theta = []

for panel in [0,1,2,3]:
    llons = np.mod(lons[panel, yrow, :], 360)
    
    theta_panel = []
    for t in range(ntime):  # Iterate over time steps
        theta_panel.append(θ[t, panel, hk, yrow, :] - T0_theta[panel, hk, yrow, :])
        print(t, np.linalg.norm(θ[t, panel, hk, yrow, :] - T0_theta[panel, hk, yrow, :]))

    all_lons.append(np.tile(llons, (ntime, 1)))
    all_times.append(np.tile(times[:, None], (1, len(llons))))
    all_theta.append(np.array(theta_panel))

# Combine arrays
combined_lons = np.concatenate(all_lons, axis=1)
combined_times = np.concatenate(all_times, axis=1)
combined_theta = np.concatenate(all_theta, axis=1)

# Contour plot with black contour lines
cont = plt.contour(combined_lons, combined_times, combined_theta,
                    levels=np.linspace(-0.1, 0.1, 4), colors='black')

# Add black contour labels
plt.clabel(cont, inline=True, fontsize=12, fmt='%1.2f', colors='blue')

# Add the thick red line between (130, 0) and (199.46, 3600)
x_line = [120, 329.88]  # Longitude values (130 to 199.46)
y_line = [0, 3600]  # Time values (0 to 3600)

x_line2 = [120, 72.088]  # Longitude values (130 to 199.46)
y_line2 = [0, 3600]  # Time values (0 to 3600)

# Plot the red line
plt.plot(x_line, y_line, color='red', linewidth=4)
plt.plot(x_line2, y_line2, color='red', linewidth=4)

# Axis labels
plt.gca().set_xlabel('Longitude', fontsize=14)
plt.gca().set_ylabel('Time (s)', fontsize=14)
xticks = [0, 120, 240, 360]
xtick_labels = [f"{x}°" for x in xticks]
plt.gca().set_xticks(xticks)
plt.gca().set_xticklabels(xtick_labels, fontsize=16)
yticks = [0, 1000, 2000, 3000]
ytick_labels = ['0 ', '1000', '2000', '3000']  # pad 0 with a space
plt.gca().set_yticks(yticks)
plt.gca().set_yticklabels(ytick_labels, fontsize=16)
plt.minorticks_on()
plt.gca().tick_params(axis='both', which='minor', direction='in', length=4, color='black',top=True, bottom=True, left=True, right=True)
plt.gca().tick_params(axis='both', which='major', direction='in', length=8, color='black',top=True, bottom=True, left=True, right=True)
# plt.gca().set_title(f'Δθ at Height Level {hk}', fontsize=14)

# Save plot
plt.savefig('DCMIP_31_Time_vs_Longitude', dpi=300, bbox_inches='tight')
plt.show()