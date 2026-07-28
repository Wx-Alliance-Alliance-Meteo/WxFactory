# WxFactory configuration options
| | | | | |
| - | - | - | - | - |
| | | | | |
 | **[General]**         | **Type**  | **Default**  | **Valid range**           | **Description**  |
   | equations             | lc-str  | [none]  | {euler, shallow_water}           |   |
   | precision             | lc-str  | double  | {double, single}                 | Runtime floating-point precision. Single precision halves state/operator storage and runtime bandwidth at the cost of accuracy; grid-dependent operators and initial metric terms are constructed in double precision before being cast.  |
   | initial_condition     | lc-str  |         |                                  |   |
   | time_start            | lc-str  |         |                                  |   |
   | time_end              | lc-str  |         |                                  |   |
   | depth_approx          | lc-str  | deep    | {deep, shallow}                  |   |
| | | | | |
 | **[System]**          | **Type**  | **Default**  | **Valid range**           | **Description**  |
   | pytorch_device        | lc-str  | cuda    | {cuda, cpu}                      | Where the computation puts its tensors. Falls back to the CPU if no GPU is available  |
| | | | | |
 | **[Test_case]**       | **Type**  | **Default**  | **Valid range**           | **Description**  |
   | case_number           | int   | -1        |                                  |   |
   | bubble_rad            | float  | 0.0      |                                  |   |
   | topography_file       | lc-str  | [none]  |                                  |   |
   | initial_conditions_file  | lc-str  | [none]  |                               |   |
   | matsuno_wave_type     | lc-str  | [none]  |                                  |   |
   | matsuno_amp           | float  | [none]   |                                  |   |
   | bubble_theta          | float  | 0.0      |                                  |   |
| | | | | |
 | **[Time_integration]**  | **Type**  | **Default**  | **Valid range**         | **Description**  |
   | dt                    | float  | [none]   | [0.0, inf]                       |   |
   | t_end                 | float  | [none]   |                                  |   |
   | time_integrator       | lc-str  | [none]  |                                  |   |
   | tolerance             | float  | 1e-07    |                                  | Convergence tolerance of the implicit / exponential integrators' inner solver. Ignored by the explicit integrators (e.g. tvdrk3), which have no inner solve.  |
   | starting_step         | int   | 0         |                                  |   |
   | exponential_solver    | lc-str  | pmex    |                                  | Registered exponential-system solver. Built-ins are pmex, pmex_ne, kiops, and exode.  |
   | exode_method          | lc-str  | bs3(2)  | {bs3(2), dp5(4), m4(3), kc3(2), exlrk3(2), exlrk4(3), f14(12), dp8(7), f10(8)}  |   |
   | exode_controller      | lc-str  |         |                                  |   |
   | krylov_size           | int   | 1         | [0, inf]                         |   |
   | krylov_mmax           | int   | 64        | [1, inf]                         | Largest Krylov subspace the exponential solver may build. The solver stores mmax+1 basis vectors of the full state, so this caps its memory; lowering it trades memory for more internal substeps.  |
   | jacobian_method       | lc-str  | complex  | {complex, fd}                   |   |
   | verbose_solver        | int   | 0         |                                  |   |
   | gmres_restart         | int   | 20        | [1, inf]                         |   |
   | splitting_integrator_1  | lc-str  |       |                                  |   |
   | splitting_integrator_2  | lc-str  |       |                                  |   |
| | | | | |
 | **[Spatial_discretization]**  | **Type**  | **Default**  | **Valid range**   | **Description**  |
   | num_elements_vertical  | int  | 1         | [1, inf]                         |   |
   | expfilter_apply       | str_to_bool  | False  |                              | Apply an exponential modal filter to the metric-weighted conservative state after each time step.  |
   | num_solpts            | int   | [none]    | [1, inf]                         |   |
   | num_elements_horizontal  | int  | [none]  | [1, inf]                         |   |
   | expfilter_order       | int   | [none]    | [2, inf]                         | Positive even order of the exponential modal filter.  |
   | expfilter_cutoff      | float  | [none]   | [0.0, 1.0]                       | Normalized modal cutoff below which modes are left unchanged.  |
   | expfilter_strength    | float  | [none]   | [0.0, inf]                       | Strength alpha in the exponential modal-filter attenuation.  |
| | | | | |
 | **[Grid]**            | **Type**  | **Default**  | **Valid range**           | **Description**  |
   | grid_file             | lc-str  |         |                                  |   |
   | lateral_boundary      | lc-str  | donor_cell  | {donor_cell, wall, periodic}  | Lateral (horizontal) boundary treatment. donor_cell exchanges traces with the neighbouring tile (the cubed sphere's inter-panel donor cell); wall reflects the tile's own boundary trace (solid wall); periodic wraps the tile to its opposite edge. The cubed sphere always uses donor_cell; wall/periodic are for cartesian grids.  |
   | discretization        | lc-str  | dg      | {dg}                             | Direct flux reconstruction / discontinuous Galerkin spatial discretization. No finite-volume RHS is currently available.  |
   | advection_only        | lc-str  | auto    | {auto, on, off}                  | Whether the Euler dynamics are frozen and only passive advection is integrated. 'auto' derives it from the DCMIP case number (cases <= 13 are advection tests); 'on'/'off' force it. Cartesian grids are always fully dynamical.  |
   | grid_type             | lc-str  | [none]  | {cubed_sphere, cartesian3d}      |   |
   | lambda0               | angle24  | [none]  |                                 | Longitude in radians of the central point of panel 0  |
   | phi0                  | angle24  | [none]  |                                 | Latitude in radians of the central point of panel 0  |
   | alpha0                | angle24  | [none]  |                                 | Rotation in radians of the central meridian of panel 0  |
   | ztop                  | float  | 0.0      |                                  |   |
   | vertical_coord        | lc-str  | gal_chen  | {gal_chen, sleve}              | Terrain-following vertical coordinate. gal_chen decays the terrain linearly with height; sleve (Schar et al. 2002) decays the large- and small-scale parts of the terrain at different rates, so that small-scale features vanish faster aloft  |
   | sleve_scale_large     | float  | 10000.0  |                                  | SLEVE decay scale height s1 of the large-scale topography (m)  |
   | sleve_scale_small     | float  | 2500.0   |                                  | SLEVE decay scale height s2 of the small-scale topography (m). Should be well below sleve_scale_large  |
   | x0                    | float  | [none]   |                                  |   |
   | x1                    | float  | [none]   |                                  |   |
   | y0                    | float  | 0.0      |                                  |   |
   | y1                    | float  | 1000.0   |                                  |   |
   | z0                    | float  | [none]   |                                  |   |
   | z1                    | float  | [none]   |                                  |   |
| | | | | |
 | **[Preconditioning]**  | **Type**  | **Default**  | **Valid range**          | **Description**  |
   | preconditioner        | lc-str  | none    | {none}                           |   |
   | verbose_precond       | int   | 0         |                                  |   |
| | | | | |
 | **[Output_options]**  | **Type**  | **Default**  | **Valid range**           | **Description**  |
   | stat_freq             | int   | 0         |                                  | Frequency in timesteps at which to print block stats  |
   | output_freq           | int   | 0         |                                  | Frequency in timesteps at which to store the solution  |
   | save_state_freq       | int   | 0         |                                  | Frequency in timesteps at which to save the state vector  |
   | output_dir            | cs-str  | results  |                                 | Directory where to store all the output  |
   | base_output_file      | cs-str  | out     |                                  | Name of file where to store the solution  |
   | store_total_time      | str_to_bool  | False  |                              | Whether to output total runtime in seconds to a file  |
   | output_format         | lc-str  | netcdf  | {netcdf, fst}                    | Cubed-sphere output format. netcdf requires netCDF4; fst additionally requires the external rmn and georef packages. Cartesian output uses its image writer regardless of this setting.  |
| | | | | |
 | **[Post_processing]**  | **Type**  | **Default**  | **Valid range**          | **Description**  |
   | enable_schar_mountain  | str_to_bool  | False  |                             | Enable Schar wave mountains to grow (stabilisation issue)  |
   | schar_mountain_longitude  | float  | [none]  |                               | Mountain longitude center point (radians)  |
   | schar_mountain_lattitude  | float  | [none]  |                               | Mountain latitude center point (radians)  |
   | schar_mountain_height  | float  | [none]  |                                  | Peak height of the mountain range (m)  |
   | schar_mountain_radius  | float  | [none]  |                                  | Mountain radius (meters)  |
   | schar_mountain_length  | float  | [none]  |                                  | Mountain wavelength (meters)  |
   | schar_mountain_step   | int   | 0         | [0, inf]                         | Number of step of interpolation, 0 is instantaneous  |
