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
#define IN_TILE_DIM 16
#define OUT_TILE_DIM (16 - 2 * NUM_OF_POINT)

__global__ void stencil_kernel(float *in, float *out, unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    float inPrev;
    __shared__ float s_inCurr[IN_TILE_DIM][IN_TILE_DIM];
    float inCurr;
    float inNext;

    if(iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev = in[(iStart - 1) * N * N + j * N + k];
    }
    if(iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr = in[iStart * N * N + j * N + k];
        s_inCurr[threadIdx.y][threadIdx.x] = inCurr;
    }

    for(int i = iStart; i < iStart + OUT_TILE_DIM; i++) {
        if(i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k <N) {
            inNext = in[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();
        if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                out[i * N * N + j * N + k] = c0 * inCurr
                                           + c1 * s_inCurr[threadIdx.y][threadIdx.x - 1]
                                           + c2 * s_inCurr[threadIdx.y][threadIdx.x + 1]
                                           + c3 * s_inCurr[threadIdx.y - 1][threadIdx.x]
                                           + c4 * s_inCurr[threadIdx.y + 1][threadIdx.x]
                                           + c5 * inPrev
                                           + c6 * inNext;
            }
        }
        __syncthreads();
        inPrev = inCurr;
        inCurr = inNext;
        s_inCurr[threadIdx.y][threadIdx.x] = inNext;
    }
}

int main() {
    int N = 64;  
    size_t size = N * N * N * sizeof(float);

    // Allocate host memory
    float *h_in = new float[N * N * N];
    float *h_out = new float[N * N * N];

    // Initialize input with random values
    for (int i = 0; i < N * N * N; ++i) {
        h_in[i] = (float)(rand() % 100);
    }

    // Allocate device memory
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, size);

    // Set up CUDA kernel launch config
    dim3 threads(IN_TILE_DIM, IN_TILE_DIM);
    dim3 blocks((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    // Launch kernel
    stencil_kernel<<<blocks, threads>>>(d_in, d_out, N);

    // Copy result back
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Verify result (optional)
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            for (int k = 1; k < N - 1; ++k) {
                int idx = i * N * N + j * N + k;
                float expected = c0 * h_in[idx]
                               + c1 * h_in[i * N * N + j * N + (k - 1)]
                               + c2 * h_in[i * N * N + j * N + (k + 1)]
                               + c3 * h_in[i * N * N + (j - 1) * N + k]
                               + c4 * h_in[i * N * N + (j + 1) * N + k]
                               + c5 * h_in[(i - 1) * N * N + j * N + k]
                               + c6 * h_in[(i + 1) * N * N + j * N + k];
                assert(fabs(expected - h_out[idx]) < 1e-3);
            }
        }
    }

    std::cout << "COMPLETED SUCCESSFULLY!" << std::endl;

    // Free memory
    delete[] h_in;
    delete[] h_out;
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
