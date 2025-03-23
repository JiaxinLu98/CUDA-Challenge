#include <iostream>
#include <cassert>

__global__ void convolution_2d(int *matrix, int *mask, int *result, int matrix_width, int matrix_height, int mask_width, int mask_height) {
    // Global thread ID calculation
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate radius of the mask
    int horizontal_r = mask_width / 2;
    int vertical_r = mask_height / 2;

    // Calculate the starting point for the element
    int horizontal_start = row - horizontal_r;
    int vertical_start = col - vertical_r;

    // Temp value for calculation
    int temp = 0;

    // Go over each element of the mask
    for(int i = 0; i < mask_width; i++) {
        for(int j = 0; j < mask_height; j++) {
            // Ignore elements that hang off (0s don't contribute)
            if(((horizontal_start + i) >= 0) && ((horizontal_start + i) < matrix_width) && ((vertical_start + j) >= 0) && ((vertical_start + j) < matrix_height)) {
                temp += mask[i * mask_width + j] * matrix[((horizontal_start + i) * matrix_width + (vertical_start + j))];
            }
        }
    }

    // Write back the results
    result[row * matrix_width + col] = temp;
}

// Initializes an n x n matrix with random numbers
// Takes:
//  m : Pointer to the matrix
//  n : Dimension of the matrix (square)
void init_matrix(int *m, int n) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        m[n * i + j] = rand() % 100;
      }
    }
}

// Verifies the 2D convolution result on the CPU
// Takes:
//  matrix:   Original matrix
//  mask:     Convolutional mask
//  result:   Result from the GPU
//  N:        Dimensions of the matrix
//  mask_dim: Dimensions of the mask
void verify_result(int *matrix, int *mask, int *result, int N, int mask_dim) {
    int radius = mask_dim / 2;

    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            int temp = 0;

            for (int i = 0; i < mask_dim; i++) {
                for (int j = 0; j < mask_dim; j++) {
                    int inRow = row - radius + i;
                    int inCol = col - radius + j;

                    if (inRow >= 0 && inRow < N && inCol >= 0 && inCol < N) {
                        temp += mask[i * mask_dim + j] * matrix[inRow * N + inCol];
                    }
                }
            }

            // Use assert to validate each result
            assert(result[row * N + col] == temp);
        }
    }
}


int main() {

    // Dimensions of the matrix (2 ^ 10 x 2 ^ 10)
    int N = 1 << 10;

    // Size of the matrix (in bytes)
    size_t bytes_n = N * N * sizeof(int);

    // Allocate the matrix and initialize it
    int *h_matrix = new int[N * N];
    int *h_result = new int[N * N];
    init_matrix(h_matrix, N);

    // Size of the mask in bytes
    int mask_dim = 7;
    size_t bytes_m = mask_dim * mask_dim * sizeof(int);

    // Allocate the mask and initialize it
    int *h_mask = new int[mask_dim * mask_dim];
    init_matrix(h_mask, mask_dim);

    // Allocate device memory
    int *d_matrix;
    int *d_mask;
    int *d_result;
    cudaMalloc(&d_matrix, bytes_n);
    cudaMalloc(&d_mask, bytes_m);
    cudaMalloc(&d_result, bytes_n);

    // Copy data to the device
    cudaMemcpy(d_matrix, h_matrix, bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, bytes_m, cudaMemcpyHostToDevice);

    // Calculate grid dimensions
    int THREADS = 16;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    // Dimension launch arguments
    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(BLOCKS, BLOCKS);

    // Perform 2D Convolution
    convolution_2d<<<grid_dim, block_dim>>>(d_matrix, d_mask, d_result, N, N, mask_dim, mask_dim);

    // Copy the result back to the CPU
    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

    // Functional test
    verify_result(h_matrix, h_mask, h_result, N, mask_dim);

    std::cout << "COMPLETED SUCCESSFULLY!" << std::endl;

    // Free the memory we allocated
    delete[] h_matrix;
    delete[] h_result;
    delete[] h_mask;

    cudaFree(d_matrix);
    cudaFree(d_result);

    return 0;
}