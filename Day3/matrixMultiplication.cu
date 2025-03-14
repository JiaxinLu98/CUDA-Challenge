#include <iostream>
#include <cassert>

__global__
void matrixMultiKernel(int *A, int *B, int *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N) {
        for(int i = 0; i < N; i++) {
            C[row * N + col] += A[row * N + i] * B[i * N + col];
        }
    }
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
    matrixMultiKernel<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, N);

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