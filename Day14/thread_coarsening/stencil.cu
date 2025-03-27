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

    __shared__ float s_inPrev[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float s_inCurr[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float s_inNext[IN_TILE_DIM][IN_TILE_DIM];

    if(iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        s_inPrev[threadIdx.y][threadIdx.x] = in[(iStart - 1) * N * N + j * N + k];
    }
    if(iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        s_inCurr[threadIdx.y][threadIdx.x] = in[iStart * N * N + j * N + k];
    }

    for(int i = iStart; i < iStart + OUT_TILE_DIM; i++) {
        if(i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k <N) {
            s_inNext[threadIdx.y][threadIdx.x] = in[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();
        if(i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if(threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                out[i * N * N + j * N + k] = c0 * s_inCurr[threadIdx.y][threadIdx.x]
                                           + c1 * s_inCurr[threadIdx.y][threadIdx.x - 1]
                                           + c2 * s_inCurr[threadIdx.y][threadIdx.x + 1]
                                           + c3 * s_inCurr[threadIdx.y - 1][threadIdx.x]
                                           + c4 * s_inCurr[threadIdx.y + 1][threadIdx.x]
                                           + c5 * s_inCurr[threadIdx.y][threadIdx.x]
                                           + c6 * s_inCurr[threadIdx.y][threadIdx.x];
            }
        }
        __syncthreads();
        s_inPrev[threadIdx.y][threadIdx.x] = s_inCurr[threadIdx.y][threadIdx.x];
        s_inCurr[threadIdx.y][threadIdx.x] = s_inNext[threadIdx.y][threadIdx.x];
    }

}