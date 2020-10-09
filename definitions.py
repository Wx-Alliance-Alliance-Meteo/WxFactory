nb_equations = 3
nbfaces     = 6

day_in_secs  = 24.0 * 3600.0    # Days in seconds

earth_radius   = 6371220.0      # Mean radius of the earth (m)
gravity        = 9.80616        # Gravitational acceleration (m s^-2)
rotation_speed = 7.29212e-5     # Angular speed of rotation of the earth (radians/s)

inv_earth_radius = 1.0 / earth_radius

p0  = 100000.     # reference pressure (Pa)
Rd  = 287.05  # J K-1 kg-1 ! gas constant for dry air
cpd = 1005.46
cvd = (cpd - Rd)  # chal. spec. air sec (volume constant) [J kg-1 K-1]

heat_capacity_ratio = cpd / cvd

# Indices for the model variables
idx_h   = 0
idx_hu1 = 1
idx_hu2 = 2
idx_u1 = 1
idx_u2 = 2
