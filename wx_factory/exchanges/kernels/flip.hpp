#ifndef FLIP_H_
#define FLIP_H_
#include "common/parameters.hpp"

template <typename T>
HOST_DEVICE_SPACE void flip_axis_kernel_shared(
    T* arr,
    size_t idx,
    size_t idx_opp // opposite
) {    
    T tmp = arr[idx];
    arr[idx] = arr[idx_opp];
    arr[idx_opp] = tmp;

}

#endif