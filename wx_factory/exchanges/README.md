

**Current issue**
- The multisize test does not pass (add_test(suite, Euler3DRestartTestCase(24, "test_multisize", optional=True), test_re))
We can run see it by running
mpirun -n 24 --bind-to none python ./tests/unit/run_mpi_tests.py --no-buffer

The relative error is around 0.0075 instead of less than 10e-15 with the original python version.
Effectively, this shows that the state vector saved if we launch the program with 6 or 24 mpi ranks.


**Where the error is**
The "main" function in the cuda implementation is *start_exchange_euler_3d*

and it calls three 
memcpy_faces_wrapper<num_t>(send_buffer, south, north, west, east, face_size);
convert_pair_wrapper_gpu<num_t, real_t>(face_data, face_boundary, send_buffer, face_size, panel, coord_size, var_size);
flip_axis_wrapper_gpu<num_t>(send_buffer, slice_shape, flip_axes, face_size, flip_flags);

*convert_pair_wrapper_gpu* causes the issue. This can be tested by commenting it.


**Likely errors**
- The error is most likely a bad indexing

- Bad memory allocation from host to device

- Bad geometric conversion
We can refer to *process_topology.py* in order to verify this

- Numerical errors. I don't think it's sufficient to justify an error this high.
However, a conversion ro real_t for instance makes the simulation explode (in the kernel pairs.hpp *convert_pair_kernel_shared*)
real_t c = (2.0 * x) / (1.0 + x*x);
// real_t c = (real_t(2) * x) / (real_t(1) + x*x); // this explodes

- 24 ranks grid adjustments?

**Call stack for the exchange kernel**
rhs_dfr.py -> start_communication # for now, we switch here between the new cuda/cpp and the original cupy implementation
process_topology.py -> start_exchange_euler_3d_cpp
wx_factory/exchanges/exchanges.cu # handles the attribution and launch of the kernels

**Plus**
Before doing the memory allocation, we asked python explicitely to make the data contiguous. Ideally, we could fuse the memory allocation with the contiguation.