#!/usr/bin/env python3

import numpy as np

# Load the arrays
Q_1 = np.load('results/state_vector_2ff905585fbc_000.00000000.npy')
Q_2 = np.load('results/state_vector_2ff905585fbc_000.00000200.npy')


# Calculate the element-wise difference
diff = np.abs(Q_2[0] - Q_1[0])

# Find the maximum difference
max_diff = np.max(diff)

# Find the maximum value of the first array (you can also choose the second array)
max_value = np.max(Q_1[0])

# Calculate the ratio of max difference to the maximum value of the first array
ratio = max_diff / max_value

# Print the result
print(f"L infinity norm: {ratio}")

