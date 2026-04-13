#include <stdio.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

using namespace cv;

// CUDA Kernel for Edge Detection
__global__ void edgeDetection(unsigned char *input, unsigned char *output, int width, int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Avoid borders
    if (x >= 1 && y >= 1 && x < width - 1 && y < height - 1) {

        int idx = y * width + x;

        int gx = input[idx + 1] - input[idx - 1];
        int gy = input[idx + width] - input[idx - width];

        int mag = abs(gx) + abs(gy);

        output[idx] = (mag > 100) ? 255 : 0;
    }
}

int main() {

    // Load image
    Mat img = imread("input/road.jpg");
    if (img.empty()) {
        printf("Error: Could not load image\n");
        return -1;
    }

    // Convert to grayscale
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    int width = gray.cols;
    int height = gray.rows;
    int size = width * height * sizeof(unsigned char);

    // Allocate GPU memory
    unsigned char *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy image to GPU
    cudaMemcpy(d_input, gray.data, size, cudaMemcpyHostToDevice);

    // Define grid and block
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    // CUDA timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch kernel
    edgeDetection<<<grid, block>>>(d_input, d_output, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("CUDA Execution Time: %f ms\n", milliseconds);

    // Copy result back
    Mat output(height, width, CV_8UC1);
    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);

    // Save output
    imwrite("output/gpu_edges.png", output);

    printf("Output saved as output/gpu_edges.png\n");

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
