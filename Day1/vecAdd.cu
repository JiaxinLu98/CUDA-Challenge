#include <iostream>
#include <cassert>

__global__
void vecAddKernel(int* A, int* B, int* C, int N) {
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Check boundary
    if(t_id < N) {
        C[t_id] = A[t_id] + B[t_id]; 
    }
}

void vecAdd(int* h_A, int* h_B, int* h_C, int N) {
    // Define array points in device
    int *d_A, *d_B, *d_C;

    // Allocate memory for d_A, d_B, d_C array in device
    size_t bytes = N * sizeof(int);
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define # of threads in a block
    int threads = 64;
    // Define # of blocks in a grid
    int blocks = (N + threads - 1) / threads;
    // Call vecAddKernel kernel
    vecAddKernel<<<blocks, threads>>>(d_A, d_B, d_C, N);

    // Copy result from device to host
    cudaMemcpy(d_C, h_C, bytes, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    // Define the length of an array (2^10 = 1024)
    int N = 1 >> 10;
    size_t bytes = N * sizeof(int);

    // Define array A, B, C pointers
    int *A, *B, *C;

    // Allocate memory for array pointers
    A = (int*)malloc(bytes);
    B = (int*)malloc(bytes);
    C = (int*)malloc(bytes);

    // Initialize A, B array
    for(int i = 0; i < N; i++){
        A[i] = (rand() % 9) + 1;
        B[i] = (rand() % 9) + 1;
    }

    // Call vecAdd() function
    vecAdd(A, B, C, N);

    // Verify the result on the GPU
    for(int i = 0; i < N; i++){
        assert(C[i] == A[i] + B[i]);
    }

    std::cout << "COMPLETED SUCCESSFULLY!\n";

    return 0;
}