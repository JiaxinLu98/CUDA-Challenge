#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstring>
#include <cuda_runtime.h>
#include <chrono>

#define BLUR_SIZE 1

// Function to get current time in seconds
double seconds() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

__global__
void warmupKernel(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0){
        printf("CUDA Warmup Done!\n");
    }
}

__global__
void blueImageKernel(unsigned char* input, unsigned char* output, int width, int height, float* kernel, int kernel_width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0;
    if(row < height && col < width){
        for(int i = -1; i <= BLUR_SIZE; i++){
            for(int j = -1; j <= BLUR_SIZE; j++){
                if((row + i) >= 0 && (col + j) >= 0){
                    sum += kernel[(i + 1) * kernel_width + (j + 1)] * input[(row + i) * width + (col + j)];
                }
            }
        }
        output[row * width + col] = sum;
    }
}



void blur_image_gpu(unsigned char* input_h, unsigned char* output_h, int width, int height, float* kernel_h, int kernel_width, int kernel_height){
    size_t size_input = width * height * sizeof(unsigned char);
    size_t size_output = size_input; 
    unsigned char *input_d, *output_d;

    size_t size_kernel = kernel_width * kernel_height * sizeof(float);
    float *kernel_d;

    // Allocate device memory for input_d, output_d, and kernel_d
    cudaMalloc((void**)&input_d, size_input);
    cudaMalloc((void**)&output_d, size_output);
    cudaMalloc((void**)&kernel_d, size_kernel);

    // Copy input and kernel to device memory
    cudaMemcpy(input_d, input_h, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_d, kernel_h, size_kernel, cudaMemcpyHostToDevice);

    // Define time measure
    double iStart, iElaps;
    cudaDeviceSynchronize();

    // **Run warmupKernel once to remove first-run overhead**
    iStart = seconds();
    warmupKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("warmup elapsed %.6f sec \n", iElaps);

    // Call kernel
    int threads = 8;
    dim3 THREADS(threads, threads);
    dim3 BLOCKS(ceil(height / threads), ceil(width / threads));
    
    // Start to measure runtime
    iStart = seconds();

    blueImageKernel<<<BLOCKS, THREADS>>>(input_d, output_d, width, height, kernel_d, kernel_width);

    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("blueImageKernel elapsed %.6f sec \n", iElaps);


    // Copy output from device memory
    cudaMemcpy(output_h, output_d, size_output, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(kernel_d);
}

int main(){

    // clock_t start, end;

    int image_width = 400;
    int image_height = 400;

    // Create a image for blurring (input image)
    cv::Mat toBlurImage(image_height, image_width, CV_8UC1);
    cv::randu(toBlurImage, cv::Scalar(0), cv::Scalar(255));
    if (cv::imwrite("./images/to_blur.jpg", toBlurImage)) {
        std::cout << "Image saved!" << std::endl;
    } else {
        std::cerr << "Error: Could not save image" << std::endl;
    }

    // Load image
    cv::Mat blurImage = cv::imread("./images/to_blur.jpg", cv::IMREAD_GRAYSCALE);

    // Check if the image is loaded successfully
    if (blurImage.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Get image dimensions
    int channels = blurImage.channels();
    size_t img_data_size = image_height * image_width * channels;
    std::cout << "Image Loaded: " << image_width << " x " << image_height
            << " with " << channels << " channel(s)\n";
    
    // Allocate buffer to store image data
    unsigned char *buffer = new unsigned char[img_data_size];

    // Copy data from to_blur.jpg to buffer
    std::memcpy(buffer, blurImage.data, img_data_size);

    // ------------------------------------------------------------------------------- //

    // Define a kernel array and store values into the array
    int kernel_width = 3;
    int kernel_height = 3;
    int kernel_size = kernel_width * kernel_height;

    // Allocate a kernel buffer to store values
    float *kernel = new float[kernel_size];

    // Store values into the kernel array
    float v = 1.0 / 9.0;
    for(int i = 0; i < kernel_size; i++){
        kernel[i] = v;
    }

    // ------------------------------------------------------------------------------- //

    // Define a output buffer array to store conmputed values
    unsigned char *output_buffer = new unsigned char[img_data_size];

    // start = clock();
    
    // Blur image
    blur_image_gpu(buffer, output_buffer, image_width, image_height, kernel, kernel_width, kernel_height);

    // ------------------------------------------------------------------------------- //
    
    // Create a blank OPenCV grayscale image 
    cv::Mat blurredImage(image_height, image_width, CV_8UC1);

    // Copy data from output_buffer to blurredImage
    std::memcpy(blurredImage.data, output_buffer, img_data_size);

    // Save the image
    cv::imwrite("./images/blurred_image.jpg", blurredImage);


    // end = clock();
    // float time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    // std::cout << "Time taken by program is: " << time_taken << " second.\n" << std::endl;

    return 0;
}
