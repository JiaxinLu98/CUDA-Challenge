// The stencil kernel assumes that each thread block is responsible for calculating a tile of output grid values
// and each threads is assigned to one pf the output frid points.

// Naive Stencil Duration: 0.5710 ms 

#include <iostream>
#include <cassert>

#define c0 0.6f
#define c1 0.066f
#define c2 0.066f
#define c3 0.066f
#define c4 0.066f
#define c5 0.066f
#define c6 0.066f

__global__ void stencil_kernel(float *in, float *out, unsigned int N) {
    // Global thread ID calculation
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    // Go over values in input 3D grid
    if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i * N * N + j * N + k] = c0 * in[i * N * N + j * N + k]
                                   + c1 * in[i * N * N + j * N + (k - 1)]
                                   + c2 * in[i * N * N + j * N + (k + 1)]
                                   + c3 * in[i * N * N + (j - 1) * N + k]
                                   + c4 * in[i * N * N + (j + 1) * N + k]
                                   + c5 * in[(i - 1) * N * N + j * N + k]
                                   + c6 * in[(i + 1) * N * N + j * N + k];
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
                if (std::fabs(reference[idx] - output[idx]) > 1e-4f) {
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
    int THREADS = 8;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    // Dimension launch argument
    dim3 block_dim(THREADS, THREADS, THREADS);
    dim3 grid_dim(BLOCKS, BLOCKS, BLOCKS);

    // Start to measure runtime
    iStart = clock();

    // Perform 3D Stencil
    stencil_kernel<<<grid_dim, block_dim>>>(d_in, d_out, N);

    cudaDeviceSynchronize();
    iElaps = ((clock() - iStart) / CLOCKS_PER_SEC) * 1000;
    printf("Naive Stencil Duration: %.4f ms \n", iElaps);

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