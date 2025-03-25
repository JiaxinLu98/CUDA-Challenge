// Tiled Duration: 0.1780 ms

#include <iostream>
#include <cassert>

#define FILTER_RADIUS 3
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

__constant__ int mask[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

__global__ void convolution_2d(int *matrix, int *result, int width, int height) {
    // Global thread ID calculation
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;

    // Load input tile (static)
    __shared__ int s_matrix[IN_TILE_DIM][IN_TILE_DIM];

    if(row >= 0 && row < height && col >= 0 && col < width) {
        s_matrix[threadIdx.y][threadIdx.x] = matrix[row * width + col];
    }
    else {
        s_matrix[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    // Calculate output elements
    int tileRow = threadIdx.y - FILTER_RADIUS;
    int tileCol = threadIdx.x - FILTER_RADIUS;

    // Turn off the threads at the edges of the block
    if(row >= 0 && row < height && col >= 0 && col < width) {
        if(tileRow >= 0 && tileRow < OUT_TILE_DIM && tileCol >= 0 && tileCol < OUT_TILE_DIM) {
            int temp = 0;

            // Go over elements in mask
            for(int i = 0; i < 2 * FILTER_RADIUS + 1; i++) {
                for(int j = 0; j < 2 * FILTER_RADIUS + 1; j++) {
                    temp += s_matrix[tileRow + i][tileCol + j] * mask[i][j];
                }
            }

            result[row * width + col] = temp;
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

void verify_result(int *matrix, int *mask, int *result, int N, int mask_dim) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            int temp = 0;

            for (int i = 0; i < mask_dim; i++) {
                for (int j = 0; j < mask_dim; j++) {
                    int inRow = row - FILTER_RADIUS + i;
                    int inCol = col - FILTER_RADIUS + j;

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

    // Size of the mask in bytes
    int mask_dim = 2 * FILTER_RADIUS + 1;
    size_t bytes_m = mask_dim * mask_dim * sizeof(int);

    // Allocate the matrix, result matrix, and mask ...
    int *h_matrix = new int[N * N];
    int *h_result = new int[N * N];
    int *h_mask = new int[mask_dim * mask_dim];

    // ... initialize the matrix and the mask
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            h_matrix[i * N + j] = rand() % 100;
        }
    }

    for(int i = 0; i < mask_dim; i++) {
        for(int j = 0; j < mask_dim; j++) {
            h_mask[i * mask_dim + j] = rand() % 10;
        }
    }

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
    int THREADS = IN_TILE_DIM;
    int BLOCKS = (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM;

    // Dimension launch arguments
    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(BLOCKS, BLOCKS);

    // Start to measure runtime
    iStart = clock();

    // Perform 2D Convolution
    convolution_2d<<<grid_dim, block_dim>>>(d_matrix, d_result, N, N);

    cudaDeviceSynchronize();
    iElaps = ((clock() - iStart) / CLOCKS_PER_SEC) * 1000;
    printf("Tiled Duration: %.4f ms \n", iElaps);

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