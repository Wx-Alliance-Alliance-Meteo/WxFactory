#!/usr/bin/env python3

import sys
import os

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
src_dir = os.path.join(root_dir, "wx_factory")
sys.path.append(root_dir)
sys.path.append(src_dir)

import netCDF4 as nc
import output
import matplotlib.pyplot as plt
import numpy as np
import output
from mpi4py import MPI
np.set_printoptions(linewidth=sys.maxsize)


data = nc.Dataset('./tests/data/temp/out.nc', 'r')
_, T0 = output.InputManager.read_config_from_save_file('./tests/data/temp/old/state_vector_47170bb8616a.00000000.npy', MPI.COMM_WORLD)

# Setup figure
fig = plt.figure(figsize=(10, 3))

times = data['time']
θ = data['theta']
elev = data['elev']
lons = data['lons']
lats = data['lats']
(ntime, npanels, nk, nj, ni) = θ.shape

# Define your data or variables (lons, elev, θ, times, nj, etc.)

T0_theta = T0[:, 4, ...] / T0[:, 0, ...]

ts = -1
yrow = nj - 1
dthetamin = np.amin(θ[ts, :, :, yrow, :] - T0_theta[:, :, yrow, :])
dthetamax = np.amax(θ[ts, :, :, yrow, :] - T0_theta[:, :, yrow, :])
abstheta = max(abs(dthetamin), abs(dthetamax))
#bstheta = min(abstheta, 3)


handles = []
all_lons = []
all_elev = []
all_theta = []
for panel in [0,1,2,3]:
    llons = np.mod(0+lons[panel,yrow,:],360)-0
    
    all_lons.append(np.tile(llons,(elev.shape[1],1)))
    all_elev.append(elev[panel,:,yrow,:]/1e3)
    all_theta.append(θ[ts,panel,:,yrow,:]-T0_theta[panel,:,yrow,:])

combined_lons = np.concatenate(all_lons,axis=1)
combined_elev = np.concatenate(all_elev,axis=1)
combined_theta = np.concatenate(all_theta,axis=1)

lin_space = np.linspace(dthetamin, dthetamax, 10)
labels = np.linspace(dthetamin, dthetamax, 10)
c = plt.contourf(combined_lons,combined_elev,combined_theta,levels=lin_space,cmap='summer')
cb = plt.colorbar(shrink=1)
cont = plt.contour(combined_lons,combined_elev,combined_theta,levels=labels, colors=['black'], linewidths= 0.8)
# cb.ax.set_ylabel('W',fontsize=14)

cb.set_ticks(ticks=labels, labels=[str(pos) for pos in labels])
cb.ax.set_ylabel('Δθ',fontsize=14)
plt.gca().set_xlabel('\phi')
plt.gca().set_xticks([0,90,180,270,360])
plt.gca().set_title('t = 3600s')
plt.gca().tick_params(axis='x', labelsize=14)
plt.gca().tick_params(axis='y', labelsize=14)
plt.gca().set_ylabel('H (km)', fontsize=14)
plt.gca().set_xlabel('Longitude', fontsize=14)
plt.savefig('DCMIP_31_3600', dpi=300, bbox_inches='tight')