#ifndef FLIP_H_
#define FLIP_H_

#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>

#include <iostream>
using std::cout;
using std::endl;


/*
Same as x.flip(array, axis=flip_dim) in python
For multiple axes, use case example:
    std::vector<int> flip_axes = {-3, -1};
    for (int ax : flip_axes) {
        flip_along_axis(data, shape, ax);
    }
*/

template <typename T>
void flip_axis(
    T* arr,
    const std::vector<int>& shape,
    const std::vector<int>& axes
) {

    const int ndim = shape.size();

    // stride per dimension
    std::vector<int> stride(ndim);
    stride[ndim-1] = 1;
    for (int d = ndim - 2; d >= 0; --d) {
        stride[d] = stride[d+1] * shape[d+1];
    }
    
    int total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

    for (int axis : axes) {

        // convert python negative format
        if (axis < 0) {
            axis += ndim;
        }

        const int dim = shape[axis];
        const int inner = stride[axis];
        const int outer = total_size / (dim * inner);


        // Clamp starting with outmost left and right and finish to center
        for (int o = 0; o < outer; ++o) {
            T* base = arr + o * dim * inner;   
            T* left = base;
            T* right = base + (dim - 1) * inner;


            while (left < right) {

                for (int j = 0; j < inner; ++j) {
                    // TODO: vectorize the memory swap, std::swap for sanity check
                    std::swap(left[j], right[j]);
                }
                left += inner;
                right -= inner;
            }

        }
    }
}

#endif