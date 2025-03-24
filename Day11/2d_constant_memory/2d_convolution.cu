// Constant Memory Duration: 0.1600 ms 

#include <iostream>
#include <cassert>

#define MASK_DIM 7

// Allocate space for the mask in constant memory
__constant__ int mask[MASK_DIM * MASK_DIM];

__global__ void convolution_2d(int *matrix, int *result, int N) {
    // Global thread ID calculation
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate radius of the mask
    int r = MASK_DIM / 2;

    // Calculate the starting point for the element
    int start_row = row - r;
    int start_col = col - r;

    // Temp value for calculation
    int temp = 0;

    // Go over each elements of the mask
    for(int i = 0; i < MASK_DIM; i++) {
        for(int j = 0; j < MASK_DIM; j++) {
            // Ignore elements that hang off
            if(((start_row + i) >= 0) && ((start_row + i) < N) && ((start_col + j) >= 0) && ((start_col + j) < N)) {
                temp += matrix[(start_row + i) * N + (start_col + j)] * mask[i * MASK_DIM + j];
            } 
        }
    }
    result[row * N + col] = temp;
}

void init_matrix(int *m, int n) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        m[n * i + j] = rand() % 100;
      }
    }
}

void verify_result(int *matrix, int *mask, int *result, int N) {
    int radius = MASK_DIM / 2;

    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            int temp = 0;

            for (int i = 0; i < MASK_DIM; i++) {
                for (int j = 0; j < MASK_DIM; j++) {
                    int inRow = row - radius + i;
                    int inCol = col - radius + j;

                    if (inRow >= 0 && inRow < N && inCol >= 0 && inCol < N) {
                        temp += mask[i * MASK_DIM + j] * matrix[inRow * N + inCol];
                    }
                }
            }

            // Use assert to validate each result
            assert(result[row * N + col] == temp);
        }
    }
}

// Warmup Kernel
__global__
void warmupKernel(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0){
        printf("CUDA Warmup Done!\n");
    }
}

int main() {
    // Dimensions of the matrix
    int N = 1 << 10;

    // Size of the matrix in bytes
    size_t bytes_n = N * N * sizeof(int);

    // Size of the mask in bytes
    size_t bytes_m = MASK_DIM * MASK_DIM * sizeof(int);

    // Allocate the matrix, result matrix, and mask ...
    int *h_matrix = new int[N * N];
    int *h_result = new int[N * N];
    int *h_mask = new int[MASK_DIM * MASK_DIM];

    // ... and initialize the matrix, and the mask
    init_matrix(h_matrix, N);
    init_matrix(h_mask, MASK_DIM);

    // Allocate device memory
    int *d_matrix, *d_result;
    cudaMalloc(&d_matrix, bytes_n);
    cudaMalloc(&d_result, bytes_n);

    // Copy data to the device
    cudaMemcpy(d_matrix, h_matrix, bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(mask, h_mask, bytes_m);

    // Define time measure
    double iStart, iElaps;
    cudaDeviceSynchronize();
    
    // **Run warmupKernel once to remove first-run overhead**
    iStart = clock();
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    iElaps = (clock() - iStart) / CLOCKS_PER_SEC;
    printf("warmup elapsed %.6f sec \n", iElaps);

    // Calculate grid dimensions
    int THREADS = 16;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    // Dimension launch arguments
    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(BLOCKS, BLOCKS);

    // Start to measure runtime
    iStart = clock();

    // Perform 2D Convolution
    convolution_2d<<<grid_dim, block_dim>>>(d_matrix, d_result, N);

    cudaDeviceSynchronize();
    iElaps = ((clock() - iStart) / CLOCKS_PER_SEC) * 1000;
    printf("Constant Memory Duration: %.4f ms \n", iElaps);


    // Copy the result back to the CPU
    cudaMemcpy(h_result, d_result, bytes_n, cudaMemcpyDeviceToHost);

    // Functional test
    verify_result(h_matrix, h_mask, h_result, N);

    std::cout << "COMPLETED SUCCESSFULLY!" << std::endl;

    // Free the memory we allocated
    delete[] h_matrix;
    delete[] h_result;
    delete[] h_mask;

    cudaFree(d_matrix);
    cudaFree(d_result);

    return 0;
}