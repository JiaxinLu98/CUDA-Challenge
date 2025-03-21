// This program implements a 1D convulution using CUDA,
// and stores the mask in constant memory

#include <iostream>
#include <cassert>
#include <ctime>

// Length of our convolutionmask
#define MASK_LENGTH 7

// Allocate space for the mask in constant memory
__constant__ int mask[MASK_LENGTH];

__global__ void convolution_1d(int *array, int *result, int n) {
    // Global thread ID calculation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate radius of the mask
    int r = MASK_LENGTH / 2;

    // Calculate the starting point for the element
    int start = tid - r;

    // Temp value for calculation
    int temp = 0;

    // Go over each element of the mask
    for(int i = 0; i < MASK_LENGTH; i++) {
        // Ignore elements that hang off (0s don't contribute)
        if(((start + i) >= 0) && ((start + i) < n)) {
            temp += array[start + i] * mask[i];
        }
    }

    // Write-back the results
    result[tid] = temp;
}



// Verify the result on the CPU
void verify_result(int *array, int *mask, int *result, int n) {
    int radius = MASK_LENGTH / 2;
    int temp;
    int start;
    for (int i = 0; i < n; i++) {
      start = i - radius;
      temp = 0;
      for (int j = 0; j < MASK_LENGTH; j++) {
        if ((start + j >= 0) && (start + j < n)) {
          temp += array[start + j] * mask[j];
        }
      }
      assert(temp == result[i]);
    }
}

int main() {
    // Number of elements in result array
    int n = 1 << 20;

    // Size of the array in bytes
    int bytes_n = n * sizeof(int);

    // Size of mask in bytes
    int bytes_m = MASK_LENGTH * sizeof(int);

    // Allocate the array (include edge elements)...
    int *h_array = new int[n];

    // ... and initialize it
    for(int i = 0; i < n; i++) {
        h_array[i] = rand() % 100;
    }

    // Allocate the mask and initialize it
    int *h_mask = new int[MASK_LENGTH];
    for(int i = 0; i < MASK_LENGTH; i++) {
        h_mask[i] = rand() % 10;
    }

    // Allocate the result array
    int *h_result = new int[n];

    // Allocate space on the device
    int *d_array, *d_result;
    cudaMalloc(&d_array, bytes_n);
    cudaMalloc(&d_result, bytes_n);

    // Copy the data to the device
    cudaMemcpy(d_array, h_array, bytes_n, cudaMemcpyHostToDevice);

    // **Copy the data directly to the symbol**
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);

    // Threads per block
    int THREADS = 256;

    // Number of blocks
    int GRID = (n + THREADS - 1) / THREADS;

    // Call the kernel
    convolution_1d<<<GRID, THREADS>>>(d_array, d_result, n);

    // Copy back the result
    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

    // Verify the result
    verify_result(h_array, h_mask, h_result, n);

    std::cout << "COMPLETED SUCCESSFULLY" << std::endl;

    return 0;
}
