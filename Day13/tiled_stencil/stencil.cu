// Assume a 3D seven-point stencil with one grid point on each side

// Tiled Stencil Duration: 1.0600 ms

#include <iostream>
#include <cassert>

#define c0 0.6f
#define c1 0.066f
#define c2 0.066f
#define c3 0.066f
#define c4 0.066f
#define c5 0.066f
#define c6 0.066f

#define NUM_OF_POINT 1
#define IN_TILE_DIM 8
#define OUT_TILE_DIM (8 - 2 * NUM_OF_POINT)

__global__ void stencil_kernel(float *in, float *out, int N) {
    // Global thread ID calculation (output grid)
    int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - NUM_OF_POINT;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - NUM_OF_POINT;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - NUM_OF_POINT;

    __shared__ float s_in[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    
    // Copy input 3D grid data to shared memory
    // Boundary check: avoid ghost cells
    if(i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        s_in[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
    }
    __syncthreads();

    if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        if(threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1 && threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
            out[i * N * N + j * N + k] = c0 * s_in[threadIdx.z][threadIdx.y][threadIdx.x]
                                       + c1 * s_in[threadIdx.z][threadIdx.y][threadIdx.x - 1]
                                       + c2 * s_in[threadIdx.z][threadIdx.y][threadIdx.x + 1]
                                       + c3 * s_in[threadIdx.z][threadIdx.y - 1][threadIdx.x]
                                       + c4 * s_in[threadIdx.z][threadIdx.y + 1][threadIdx.x]
                                       + c5 * s_in[threadIdx.z - 1][threadIdx.y][threadIdx.x]
                                       + c6 * s_in[threadIdx.z + 1][threadIdx.y][threadIdx.x];
        }
    }
}

void verify_result(float *input, float *output, int N) {

    float* reference = new float[N * N * N];

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                int idx = i * N * N + j * N + k;

                if (i >= 1 && i < N - 1 &&
                    j >= 1 && j < N - 1 &&
                    k >= 1 && k < N - 1) {
                    reference[idx] =
                        c0 * input[idx] +
                        c1 * input[i * N * N + j * N + (k - 1)] +
                        c2 * input[i * N * N + j * N + (k + 1)] +
                        c3 * input[i * N * N + (j - 1) * N + k] +
                        c4 * input[i * N * N + (j + 1) * N + k] +
                        c5 * input[(i - 1) * N * N + j * N + k] +
                        c6 * input[(i + 1) * N * N + j * N + k];
                } else {
                    reference[idx] = 0.0f;  // Or whatever boundary rule you apply
                }

                // Assert that GPU result matches CPU result (within tolerance)
                if (std::fabs(reference[idx] - output[idx]) > 1e-1f) {
                    std::cerr << "Mismatch at (" << i << "," << j << "," << k << "): "
                              << "Expected " << reference[idx]
                              << ", but got " << output[idx] << "\n";
                    assert(false);
                }
            }
        }
    }
    delete[] reference;
}

__global__ void warmupKernel(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0){
        printf("CUDA Warmup Done!\n");
    }
}


int main() {
    // Dimensions of the grid
    int N = 256;

    // Size of the grid (in bytes)
    size_t bytes = N * N * N  * sizeof(float);

    // Allocate the input grid and output grid ...
    float *h_in = new float[N * N * N];
    float *h_out = new float[N * N * N];

    // ... and initialize the input grid
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            for(int k = 0; k < N; k++) {
                h_in[i * N * N + j * N + k] = (float)(rand() % 100);
            }
        }
    }

    // Allocate device memory
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // Copy data to the device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, bytes);

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

    // Dimension launch argument
    dim3 block_dim(THREADS, THREADS, THREADS);
    dim3 grid_dim(BLOCKS, BLOCKS, BLOCKS);

    // Start to measure runtime
    iStart = clock();

    // Perform 3D Stencil
    stencil_kernel<<<grid_dim, block_dim>>>(d_in, d_out, N);

    cudaDeviceSynchronize();
    iElaps = ((clock() - iStart) / CLOCKS_PER_SEC) * 1000;
    printf("Tiled Stencil Duration: %.4f ms \n", iElaps);

    // Copy the result back to the CPU
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Verify the result
    verify_result(h_in, h_out, N);

    std::cout << "COMPLETED SUCCESSFULLY!" << std::endl;

    // Free the memory we allocated
    delete[] h_in;
    delete[] h_out;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}