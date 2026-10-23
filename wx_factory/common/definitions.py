nbfaces = 6

day_in_secs = 24.0 * 3600.0

gravity = 9.80616  # Gravitational acceleration (m s^-2)

p0 = 100000.0  # reference pressure (Pa)
Rd = 287.05  # J K-1 kg-1 ! gas constant for dry air
cpd = 1005.46
cvd = cpd - Rd  # chal. spec. air sec (volume constant) [J kg-1 K-1]
kappa = Rd / cpd
heat_capacity_ratio = cpd / cvd

idx_h = 0
idx_u1 = 1
idx_u2 = 2

# Euler state: density, relative covariant velocity, potential-temperature density.
idx_rho = 0
idx_v1 = 1
idx_v2 = 2
idx_v3 = 3
idx_rho_theta = 4
