
#include <iostream>

#include <cuda.h>

#define cudaCheck(cmd) cudaApiAssert((cmd), __FILE__, __LINE__)

//! Test the given return code and abort the program with an error message if it is not
//! "cudaSuccess"
inline void cudaApiAssert(const cudaError_t code, const char* filename, const int line) {
  if (code != cudaSuccess)
  {
    std::cerr << "CUDA API call failed: " << cudaGetErrorString(code) << ", at "
              << filename << ":" << line << "\n";
    exit(-1);
  }
}

//! Basic kernel to set a value for the first 10 entries of the given device array
__global__ void test_kernel(int* result) {
  const int thread_id = threadIdx.x;
  // printf("Hello, world from the device!\n");
  if (thread_id < 10)
  {
    result[thread_id] = thread_id;
  }
}

int main(void) {

  int* result_d;
  cudaCheck(cudaMalloc(&result_d, 10 * sizeof(int)));      // Allocate on device
  cudaCheck(cudaMemset(result_d, 0xff, 10 * sizeof(int))); // Initialize to -1 everywhere

  test_kernel<<<1, 10>>>(result_d);   // Launch the kernel
  cudaCheck(cudaDeviceSynchronize()); // Wait for the kernel to finish

  // Copy the result to CPU mem
  int result_h[10];
  cudaCheck(cudaMemcpy(result_h, result_d, 10 * sizeof(int), cudaMemcpyDeviceToHost));

  // Check if kernel was run properly
  int num_errors = 0;
  for (int i = 0; i < 10; i++)
  {
    std::cout << result_h[i] << " ";
    if (result_h[i] != i)
    {
      num_errors++;
    }
  }
  std::cout << std::endl;

  return num_errors;
}
