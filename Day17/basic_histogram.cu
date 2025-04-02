#include <iostream>

__global__ void histo_kernel(char *data, int length, int *histo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < length) {
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&(histo[alphabet_position / 4]), 1);
        }
    }
}

int main() {
    const char *input = "programming massively parallel processors";

    // Define the size of the data and the histogram
    int data_size = strlen(input);
    int histo_size = 7;


    // Define the size of the data and the histogram in bytes
    size_t byte_d = data_size * sizeof(char);
    size_t byte_h = histo_size * sizeof(int);

    // Allocate the data on the host
    char *h_data = new char[data_size];
    int *h_histo = new int[histo_size];
    memcpy(h_data, input, data_size);

    // Allocate the data and the histogram on the device
    char *d_data;
    int *d_histo;
    cudaMalloc(&d_data, byte_d);
    cudaMalloc(&d_histo, byte_h);

    // Copy the data from the host to the device
    cudaMemcpy(d_data, h_data, byte_d, cudaMemcpyHostToDevice);

    // Thread and block configuration
    int THREADS = 64;
    int BLOCKS = (data_size + THREADS - 1) / THREADS;

    histo_kernel<<<BLOCKS, THREADS>>>(d_data, data_size, d_histo);

    // Copy the histogram from the device to the host
    cudaMemcpy(h_histo, d_histo, byte_h, cudaMemcpyDeviceToHost);

    // Print the histogram
    for(int i = 0; i < histo_size; i++) {
        std::cout << "Histogram[" << i << "] = " << h_histo[i] << std::endl;
    }

    // Free the memory on the device
    cudaFree(d_data);
    cudaFree(d_histo);

    // Free the memory on the host
    delete[] h_data;
    delete[] h_histo;

    return 0;
}
