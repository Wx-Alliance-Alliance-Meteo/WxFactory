#ifndef PAIRS_H_
#define PAIRS_H_

#include <vector>
#include <numeric>
#include <functional>
#include <algorithm>

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
    const std::vector<int> shape&,
    const std::vector<int> axes&
) {

    // assume no python - style negatives

    const int ndim = shape.size();

    // stride per dimension
    std::vector<int> stride(ndim);
    stride[ndim-1] = 1;
    for (int d = ndim - 2; d >= 0; --d) {
        stride[d] = stride[d+1] * shape[d+1];
    }


    int total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());

    for (int axis : axes) {

        const int dim = shape[axis];
        const int mid = dim / 2;
        const int inner = stride[axis];
        const int outer = total_size / (shape[axis] * inner);

        // Clamp starting with outmost left and right and finish to center
        for (int o = 0; o < outer; ++o) {
            T* base = arr + o * dim * inner;
            for (int i = 0; i < mid; ++i) {
                T* left = base + i * inner;
                T* right = base + (shape[axis] - i - 1) * inner;
                for (int j = 0; j < inner; ++j) {
                    std::swap(left[j], right[j]);
                }
            }
        }
    }
}