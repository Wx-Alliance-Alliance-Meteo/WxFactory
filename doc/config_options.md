# WxFactory configuration options

| | | | | |
| - | - | - | - | - |
| | | | | |
 | **[General]**         | **Type**  | **Default**  | **Valid range**           | **Description**  | 
   | equations             | str   | [none]    | {euler, shallow_water}           |   | 
   | depth_approx          | str   | deep      | {deep, shallow}                  |   | 
| | | | | |
 | **[System]**          | **Type**  | **Default**  | **Valid range**           | **Description**  | 
   | desired_device        | str   | cpu       | {cpu, cuda}                      | Physical backend to run the computation on  | 
   | cuda_devices          | List  | []        |                                  | List of Nvidia physical device to use  | 
| | | | | |
 | **[Test_case]**       | **Type**  | **Default**  | **Valid range**           | **Description**  | 
   | case_number           | int   | -1        |                                  |   | 
   | matsuno_wave_type     | str   | [none]    |                                  |   | 
   | matsuno_amp           | float  | [none]   |                                  |   | 
   | bubble_theta          | float  | 0.0      |                                  |   | 
   | bubble_rad            | float  | 0.0      |                                  |   | 
| | | | | |
 | **[Time_integration]**  | **Type**  | **Default**  | **Valid range**         | **Description**  | 
   | dt                    | float  | [none]   | [0.0, inf]                       |   | 
   | t_end                 | float  | [none]   |                                  |   | 
   | time_integrator       | str   | [none]    |                                  |   | 
   | tolerance             | float  | [none]   |                                  |   | 
   | starting_step         | int   | 0         |                                  |   | 
   | exponential_solver    | str   | pmex      | {pmex, kiops, exode, pmex_ne, cwy_1s, cwy_ne, cwy_ne1s, dcgs2, icwy_1s, icwy_neiop, icwy_ne, icwy_ne1s, kiops_ne, pmex_1s, pmex_ne1s}  |   | 
   | exode_method          | str   | bs3(2)    | {bs3(2), dp5(4), m4(3), kc3(2), exlrk3(2), exlrk4(3), f14(12), dp8(7), f10(8)}  |   | 
   | exode_controller      | str   |           |                                  |   | 
   | krylov_size           | int   | 1         | [0, inf]                         |   | 
   | jacobian_method       | str   | complex   | {complex, fd}                    |   | 
   | linear_solver         | str   | fgmres    | {fgmres, gcrot}                  |   | 
   | verbose_solver        | int   | 0         |                                  |   | 
   | gmres_restart         | int   | 20        | [1, inf]                         |   | 
| | | | | |
 | **[Spatial_discretization]**  | **Type**  | **Default**  | **Valid range**   | **Description**  | 
   | num_solpts            | int   | [none]    | [1, inf]                         |   | 
   | num_elements_horizontal  | int  | [none]  | [1, inf]                         |   | 
   | num_elements_vertical  | int  | 1         | [1, inf]                         |   | 
   | filter_apply          | bool  | False     |                                  |   | 
   | filter_order          | int   | 16        |                                  |   | 
   | filter_order          | int   | 0         |                                  |   | 
   | filter_cutoff         | float  | 0.25     |                                  |   | 
   | filter_cutoff         | float  | 0.0      |                                  |   | 
   | expfilter_apply       | bool  | False     |                                  |   | 
   | expfilter_order       | int   | [none]    |                                  |   | 
   | expfilter_order       | int   | 0         |                                  |   | 
   | expfilter_cutoff      | float  | [none]   |                                  |   | 
   | expfilter_cutoff      | float  | 0.0      |                                  |   | 
   | expfilter_strength    | float  | [none]   |                                  |   | 
   | expfilter_strength    | float  | 0.0      |                                  |   | 
   | apply_sponge          | bool  | False     |                                  |   | 
   | sponge_tscale         | float  | 1.0      |                                  |   | 
   | sponge_zscale         | float  | 0.0      |                                  |   | 
| | | | | |
 | **[Grid]**            | **Type**  | **Default**  | **Valid range**           | **Description**  | 
   | grid_type             | str   | [none]    | {cubed_sphere, cartesian2d}      |   | 
   | discretization        | str   | dg        | {dg, fv}                         |   | 
   | lambda0               | float  | [none]   |                                  |   | 
   | phi0                  | float  | [none]   |                                  |   | 
   | alpha0                | float  | [none]   |                                  |   | 
   | ztop                  | float  | 0.0      |                                  |   | 
   | x0                    | float  | [none]   |                                  |   | 
   | x1                    | float  | [none]   |                                  |   | 
   | z0                    | float  | [none]   |                                  |   | 
   | z1                    | float  | [none]   |                                  |   | 
| | | | | |
 | **[Preconditioning]**  | **Type**  | **Default**  | **Valid range**          | **Description**  | 
   | preconditioner        | str   | none      | {none, fv, fv-mg, p-mg, lu, ilu}  |   | 
   | precond_flux          | str   | ausm      | {ausm, upwind, rusanov}          |   | 
   | num_mg_levels         | int   | 1         | [1, inf]                         |   | 
   | precond_tolerance     | float  | 0.1      |                                  |   | 
   | num_pre_smoothe       | int   | 1         | [0, inf]                         |   | 
   | num_post_smoothe      | int   | 1         | [0, inf]                         |   | 
   | mg_smoother           | str   | exp       | {exp, kiops, erk3, erk1, ark3}   |   | 
   | exp_smoothe_spectral_radii  | List  | [2.0]  |                               |   | 
   | exp_smoothe_num_iters  | List  | [4]      |                                  |   | 
   | mg_solve_coarsest     | bool  | False     |                                  |   | 
   | kiops_dt_factor       | float  | 1.1      |                                  |   | 
   | verbose_precond       | int   | 0         |                                  |   | 
   | dg_to_fv_interp       | str   | lagrange  | {l2-norm, lagrange}              |   | 
   | pseudo_cfl            | float  | 1.0      |                                  |   | 
| | | | | |
 | **[Output_options]**  | **Type**  | **Default**  | **Valid range**           | **Description**  | 
   | stat_freq             | int   | 0         |                                  | Frequency in timesteps at which to print block stats  | 
   | output_freq           | int   | 0         |                                  | Frequency in timesteps at which to store the solution  | 
   | save_state_freq       | int   | 0         |                                  | Frequency in timesteps at which to save the state vector  | 
   | store_solver_stats    | bool  | False     |                                  | Whether to store solver stats (at every timestep)  | 
   | output_dir            | cs-str  | results  |                                 | Directory where to store all the output  | 
   | base_output_file      | cs-str  | out     |                                  | Name of file where to store the solution  | 
   | solver_stats_file     | cs-str  | solver_stats.db  |                         | SQL file where to store statistics for this run  | 
   | store_total_time      | bool  | False     |                                  | Whether to output total runtime in seconds to a file  | 
   | output_format         | str   | netcdf    |                                  | Desired format to use for storing simulation results.  | 
