#ifndef MEM_CPY_FACES
#define MEM_CPY_FACES

#include "common/parameters.hpp"

// Not used but left here for reference
template <typename num_t>
HOST_DEVICE_SPACE void memcpy_faces_kernel_legacy(
    num_t* send_buffer,
    const num_t* south,
    const num_t* north,
    const num_t* west,
    const num_t* east,
    size_t face_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t face_id = blockIdx.y;

    if (idx < face_size) {
        const num_t* src;
        switch (face_id) {
            case 0: src = south; break;
            case 1: src = north; break;
            case 2: src = west;  break;
            case 3: src = east;  break;
        }
        send_buffer[face_id * face_size + idx] = src[idx];
    }
}

#endif // MEM_CPY_FACES