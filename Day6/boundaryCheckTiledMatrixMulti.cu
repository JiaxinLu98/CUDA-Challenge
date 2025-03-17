// Allow the kernel to correctly handle matrices whose width is not a multiple of the tile width

#include <iostream>
#include <cassert>

#define TILE_WIDTH 15

__global__
void matrixMultiKernel(int *A, int *B, int *C, int N, int phases) {
    __shared__ int Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Bds[TILE_WIDTH][TILE_WIDTH];

    // Reside in registers (scope: individual thread)
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identify the row and column of the C element to work on 
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // Loop over the A and B tiles required to compute C element
    int Cvalue = 0;
    // The ph variable indicates the number of phases that have already been done
    for(int ph = 0; ph < phases; ph++) {
        // Collaborative loading of A and B tiles into shared memory
        if(row < N && (ph * TILE_WIDTH + tx) < N) {
            Ads[ty][tx] = A[row * N + ph * TILE_WIDTH + tx];    // row index: row; column index: ph*TILE_WIDTH + tx;
        }
        else Ads[ty][tx] = 0;

        if((ph * TILE_WIDTH + ty) < N && col < N) {
            Bds[ty][tx] = B[(ph * TILE_WIDTH + ty) * N + col];  // row index: ph * TILE_WIDTH + ty; column index: col;
        }
        else Bds[ty][tx] = 0;

        // Make sure all elements are loaded into the shared memory before using them
        // read-after-write dependence: threads must wait for data to be written to the proper place before they try to read it
        __syncthreads();

        for(int i = 0; i < TILE_WIDTH; i++) {
            Cvalue += Ads[ty][i] * Bds[i][tx];
        }
        
        // Make sure all threads have finished using the A and B elements in the shares memory
        // write-after-read dependence: a thread must wait for the data to be read by all threads
        __syncthreads();
    }

    C[row * N + col] = Cvalue;
}

void init_matrix(int* matrix, int N) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            matrix[i * N + j] = rand() % 9 + 1;
        }
    }
}

void matrixMulti(int* h_A, int* h_B, int* h_C, int N) {
    int *d_A, *d_B, *d_C;
    size_t bytes = N * N * sizeof(int);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int threads = 64;
    int blocks = (N + threads - 1) / threads;
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(blocks, blocks);
    int phases = ceil(N / TILE_WIDTH);
    matrixMultiKernel<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, N, phases);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void verify_result(int* A, int* B, int* C, int N) {
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        int tmp = 0;
        for (int k = 0; k < N; k++) {
          tmp += A[i * N + k] * B[k * N + j];
        }
        assert(tmp == C[i * N + j]);
      }
    }
  }

int main() {
    int N = 1 >> 10;
    size_t bytes = N * N * sizeof(int);

    int *h_A, *h_B, *h_C;
    h_A = (int*)malloc(bytes);
    h_B = (int*)malloc(bytes);
    h_C = (int*)malloc(bytes);

    init_matrix(h_A, N);
    init_matrix(h_B, N);

    matrixMulti(h_A, h_B, h_C, N);

    verify_result(h_A, h_B, h_C, N);

    std::cout << "COMPLETED SUCCESSFULLY" << std::endl;

    return 0;
}
