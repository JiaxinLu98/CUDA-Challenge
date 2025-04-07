#include <iostream>

#define NUM_BINS 7

__global__ void histo_private_kernel(char *data, int length, int *histo) {
    // Initialize privated bins
    __shared__ int s_histo[NUM_BINS];
    for(int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        s_histo[bin] = 0u;
    }
    __syncthreads();

    // Histogram
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < length) {
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(s_histo[alphabet_position / 4]), 1);
        } 
    }
    __syncthreads();

    // Commit to global memory
    for(int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        int binValue = s_histo[bin];
        if(binValue > 0) {
            atomicAdd(&(histo[bin]), binValue);
        }
    }
}
