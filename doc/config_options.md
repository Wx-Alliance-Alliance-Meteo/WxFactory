# WxFactory configuration options
| | | | | |
| - | - | - | - | - |
| | | | | |
 | **[General]**         | **Type**  | **Default**  | **Valid range**           | **Description**  | 
   | equations             | lc-str  | [none]  | {euler, shallow_water}           |   | 
   | depth_approx          | lc-str  | deep    | {deep, shallow}                  |   | 
| | | | | |
 | **[System]**          | **Type**  | **Default**  | **Valid range**           | **Description**  | 
   | desired_device        | lc-str  | cpp     | {cpp, cuda, cupy, numpy, omp}    | Physical backend to run the computation on  | 
   | cuda_devices          | int   | []        |                                  | List of Nvidia physical device to use  | 
| | | | | |
 | **[Test_case]**       | **Type**  | **Default**  | **Valid range**           | **Description**  | 
   | case_number           | int   | -1        |                                  |   | 
   | bubble_theta          | float  | 0.0      |                                  |   | 
   | bubble_rad            | float  | 0.0      |                                  |   | 
   | topography_file       | lc-str  | [none]  |                                  |   | 
   | initial_conditions_file  | lc-str  | [none]  |                               |   | 
   | matsuno_wave_type     | lc-str  | [none]  |                                  |   | 
   | matsuno_amp           | float  | [none]   |                                  |   | 
| | | | | |
 | **[Time_integration]**  | **Type**  | **Default**  | **Valid range**         | **Description**  | 
   | dt                    | float  | [none]   | [0.0, inf]                       |   | 
   | t_end                 | float  | [none]   |                                  |   | 
   | time_integrator       | lc-str  | [none]  |                                  |   | 
   | tolerance             | float  | [none]   |                                  |   | 
   | starting_step         | int   | 0         |                                  |   | 
   | exponential_solver    | lc-str  | pmex    | {pmex, kiops, exode, pmex_ne, cwy_1s, cwy_ne, cwy_ne1s, dcgs2, icwy_1s, icwy_neiop, icwy_ne, icwy_ne1s, kiops_ne, pmex_1s, pmex_ne1s}  |   | 
   | exode_method          | lc-str  | bs3(2)  | {bs3(2), dp5(4), m4(3), kc3(2), exlrk3(2), exlrk4(3), f14(12), dp8(7), f10(8)}  |   | 
   | exode_controller      | lc-str  |         |                                  |   | 
   | krylov_size           | int   | 1         | [0, inf]                         |   | 
   | jacobian_method       | lc-str  | complex  | {complex, fd}                   |   | 
   | verbose_solver        | int   | 0         |                                  |   | 
   | gmres_restart         | int   | 20        | [1, inf]                         |   | 
   | splitting_integrator_1  | lc-str  |       |                                  |   | 
   | splitting_integrator_2  | lc-str  |       |                                  |   | 
| | | | | |
 | **[Spatial_discretization]**  | **Type**  | **Default**  | **Valid range**   | **Description**  | 
   | num_elements_vertical  | int  | 1         | [1, inf]                         |   | 
   | filter_apply          | str_to_bool  | False  |                              |   | 
   | expfilter_apply       | str_to_bool  | False  |                              |   | 
   | num_solpts            | int   | [none]    | [1, inf]                         |   | 
   | num_elements_horizontal  | int  | [none]  | [1, inf]                         |   | 
   | filter_order          | int   | 16        |                                  |   | 
   | filter_order          | int   | 0         |                                  |   | 
   | filter_cutoff         | float  | 0.25     |                                  |   | 
   | filter_cutoff         | float  | 0.0      |                                  |   | 
   | expfilter_order       | int   | [none]    |                                  |   | 
   | expfilter_order       | int   | 0         |                                  |   | 
   | expfilter_cutoff      | float  | [none]   |                                  |   | 
   | expfilter_cutoff      | float  | 0.0      |                                  |   | 
   | expfilter_strength    | float  | [none]   |                                  |   | 
   | expfilter_strength    | float  | 0.0      |                                  |   | 
| | | | | |
 | **[Grid]**            | **Type**  | **Default**  | **Valid range**           | **Description**  | 
   | grid_file             | lc-str  |         |                                  |   | 
   | discretization        | lc-str  | dg      | {dg, fv}                         |   | 
   | grid_type             | lc-str  | [none]  | {cubed_sphere, cartesian3d}      |   | 
   | lateral_boundary      | lc-str  | donor_cell | {donor_cell, wall, periodic}  | Horizontal boundary treatment. donor_cell for the cubed sphere; wall/periodic for cartesian3d  | 
   | advection_only        | lc-str  | auto    | {auto, on, off}                  | Freeze the dynamics and advect passively. auto = derive from the DCMIP case number  | 
   | lambda0               | angle24  | [none]  |                                 | Longitude in radians of the central point of panel 0  | 
   | phi0                  | angle24  | [none]  |                                 | Latitude in radians of the central point of panel 0  | 
   | alpha0                | angle24  | [none]  |                                 | Rotation in radians of the central meridian of panel 0  | 
   | ztop                  | float  | 0.0      |                                  |   | 
   | x0                    | float  | [none]   |                                  | Cartesian (x, z) box, extruded as a thin y-slab for cartesian3d  | 
   | x1                    | float  | [none]   |                                  |   | 
   | y0                    | float  | 0.0      |                                  | y-extent of the (empty) extruded dimension for cartesian3d  | 
   | y1                    | float  | 1000.0   |                                  |   | 
   | z0                    | float  | [none]   |                                  |   | 
   | z1                    | float  | [none]   |                                  |   | 
| | | | | |
 | **[Preconditioning]**  | **Type**  | **Default**  | **Valid range**          | **Description**  | 
   | preconditioner        | lc-str  | none    | {none, fv, fv-mg, p-mg, lu, ilu}  |   | 
   | precond_flux          | lc-str  | ausm    | {ausm, upwind, rusanov}          |   | 
   | precond_tolerance     | float  | 0.1      |                                  |   | 
   | num_pre_smoothe       | int   | 1         | [0, inf]                         |   | 
   | num_post_smoothe      | int   | 1         | [0, inf]                         |   | 
   | mg_smoother           | lc-str  | exp     | {exp, kiops, erk3, erk1, ark3}   |   | 
   | mg_solve_coarsest     | str_to_bool  | False  |                              |   | 
   | kiops_dt_factor       | float  | 1.1      |                                  |   | 
   | verbose_precond       | int   | 0         |                                  |   | 
   | dg_to_fv_interp       | lc-str  | lagrange  | {l2-norm, lagrange}            |   | 
   | pseudo_cfl            | float  | 1.0      |                                  |   | 
   | num_mg_levels         | int   | 1         | [1, inf]                         |   | 
   | exp_smoothe_spectral_radii  | float  | [2.0]  |                              |   | 
   | exp_smoothe_num_iters  | int  | [4]       |                                  |   | 
| | | | | |
 | **[Output_options]**  | **Type**  | **Default**  | **Valid range**           | **Description**  | 
   | stat_freq             | int   | 0         |                                  | Frequency in timesteps at which to print block stats  | 
   | output_freq           | int   | 0         |                                  | Frequency in timesteps at which to store the solution  | 
   | save_state_freq       | int   | 0         |                                  | Frequency in timesteps at which to save the state vector  | 
   | store_solver_stats    | str_to_bool  | False  |                              | Whether to store solver stats (at every timestep)  | 
   | output_dir            | cs-str  | results  |                                 | Directory where to store all the output  | 
   | base_output_file      | cs-str  | out     |                                  | Name of file where to store the solution  | 
   | solver_stats_file     | cs-str  | solver_stats.db  |                         | SQL file where to store statistics for this run  | 
   | store_total_time      | str_to_bool  | False  |                              | Whether to output total runtime in seconds to a file  | 
   | output_format         | lc-str  | netcdf  |                                  | Desired format to use for storing simulation results.  | 
| | | | | |
 | **[Post_processing]**  | **Type**  | **Default**  | **Valid range**          | **Description**  | 
   | enable_schar_mountain  | str_to_bool  | False  |                             | Enable Schar wave mountains to grow (stabilisation issue)  | 
   | schar_mountain_longitude  | float  | [none]  |                               | Mountain longitude center point (radians)  | 
   | schar_mountain_lattitude  | float  | [none]  |                               | Mountain latitude center point (radians)  | 
   | schar_mountain_height  | float  | [none]  |                                  | Peak height of the mountain range (m)  | 
   | schar_mountain_radius  | float  | [none]  |                                  | Mountain radius (meters)  | 
   | schar_mountain_length  | float  | [none]  |                                  | Mountain wavelength (meters)  | 
   | schar_mountain_step   | int   | 0         | [0, inf]                         | Number of step of interpolation, 0 is instantaneous  | 
