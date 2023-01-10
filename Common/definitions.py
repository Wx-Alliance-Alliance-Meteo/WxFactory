nbfaces     = 6

day_in_secs  = 24.0 * 3600.0    # Days in seconds

gravity      = 9.80616        # Gravitational acceleration (m s^-2)

p0  = 100000.     # reference pressure (Pa)
Rd  = 287.05  # J K-1 kg-1 ! gas constant for dry air
cpd = 1005.46
cvd = (cpd - Rd)  # chal. spec. air sec (volume constant) [J kg-1 K-1]
kappa = Rd / cpd
heat_capacity_ratio = cpd / cvd

# Indices for the shallow water model variables
idx_h   = 0
idx_hu1 = 1
idx_hu2 = 2
idx_u1 = 1
idx_u2 = 2

# Indices for the Euler model variables
idx_rho       = 0
idx_rho_u1    = 1
idx_rho_u2    = 2
idx_rho_w     = 3
idx_rho_theta = 4

# Indices for the cartesian 2D grid (Euler)
idx_2d_rho       = 0
idx_2d_rho_u     = 1
idx_2d_rho_w     = 2
idx_2d_rho_theta = 3
