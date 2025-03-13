#include <stdio.h>

#define CHANNELS 3

// Warmup Kernel
__global__
void warmupKernel(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0){
        printf("CUDA Warmup Done!\n");
    }
}

// Convert color image to grayscale L = 0.21 * r + 0.72 * g + 0.07 * b
__global__
void imageConversionKernel(unsigned char* Pin, unsigned char* Pout, int width, int height){
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(col < width && row < height){
        // Get 1D offset for the grayscale image
        int gray_offset = row * width + col;
        // Get 1D offset for the RGB image
        int rgb_offset = gray_offset * CHANNELS;
        unsigned char r = Pin[rgb_offset + 2];
        unsigned char g = Pin[rgb_offset + 1];
        unsigned char b = Pin[rgb_offset];
        Pout[gray_offset] = 0.21 * r + 0.72 * g + 0.07 * b;
    }

}

void imageConversion(unsigned char* Pin_h, unsigned char* Pout_h, int width, int height){
    int size_Pin = width * height * 3 * sizeof(unsigned char);
    int size_Pout = width * height * sizeof(unsigned char);
    unsigned char *Pin_d, *Pout_d;

    // Part 1: Allocate device memory for Pin and Pout
    // Copy Pin to device memory
    cudaMalloc((void**)&Pin_d, size_Pin);
    cudaMalloc((void**)&Pout_d, size_Pout);

    cudaMemcpy(Pin_d, Pin_h, size_Pin, cudaMemcpyHostToDevice);

    // Define time measure
    double iStart, iElaps;
    cudaDeviceSynchronize();

    // **Run warmupKernel once to remove first-run overhead**
    iStart = clock();
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    iElaps = (clock() - iStart) / CLOCKS_PER_SEC;
    printf("warmup elapsed %.6f sec \n", iElaps);

    // Part 2: Call kernel - to launch a grid of threads to perform the conversion
    int threads = 16;
    dim3 THREADS(threads, threads);
    dim3 Blocks(ceil(width / threads), ceil(height / threads));

    // Start to measure runtime
    iStart = clock();

    imageConversionKernel<<<Blocks, THREADS>>>(Pin_d, Pout_d, width, height);

    cudaDeviceSynchronize();
    iElaps = (clock() - iStart) / CLOCKS_PER_SEC;
    printf("imageConversionKernel elapsed %.6f sec \n", iElaps);

    // Part 3: Copy Pout from the device memory
    // Free device vectors
    cudaMemcpy(Pout_h, Pout_d, size_Pout, cudaMemcpyDeviceToHost);

    cudaFree(Pin_d);
    cudaFree(Pout_d);
}

int main(){

    // Define a File Pointer
    FILE *fIn;
    FILE *fOut;

    // Open the File
    fIn = fopen("lena_color.bmp", "r");
    fOut = fopen("lena_gray.bmp", "w+");

    if(!fIn){
        perror("Failed to open file");
        return 1;
    }

    // Strip out the image header
    unsigned char image_header[54];
    for(int i = 0; i < 54; i++){
        image_header[i] = getc(fIn);
    }

    // Write the image header into lena_gray.bmp
    fwrite(image_header, sizeof(unsigned char), 54, fOut);

    // Extract image height, width
    int height = *(int*)&image_header[18];
    int width = *(int*)&image_header[22];
    printf("width: %d\n",width);
    printf("height: %d\n",height);

    // Define arrays Pin, Pout pointers
    unsigned char *Pin, *Pout;

    // Memory allocation for arrays Pin and Pout
    Pin = (unsigned char*)malloc(width * height * CHANNELS * sizeof(unsigned char));
    Pout = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    // Initialize array Pin
    for(int i = 0; i < width * height * CHANNELS; i += 3){
        Pin[i] = getc(fIn);      // B
        Pin[i + 1] = getc(fIn);  // G
        Pin[i + 2] = getc(fIn);  // R
    }

    imageConversion(Pin, Pout, width, height);

    for(int i = 0; i < width * height; i++){
        putc(Pout[i], fOut);
        putc(Pout[i], fOut);
        putc(Pout[i], fOut);
    }

    // 8. Close the File
    fclose(fIn);
    fclose(fOut);

    return 0;
}