#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {

    // Load image
    Mat img = imread("road.jpg");
    if (img.empty()) {
        printf("Error loading image\n");
        return -1;
    }

    int width = img.cols;
    int height = img.rows;

    // Convert to grayscale
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Create output image
    Mat output(height, width, CV_8UC1);

    clock_t start = clock();

    // Edge detection (manual gradient)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {

            int idx = y * width + x;

            int gx = gray.data[idx + 1] - gray.data[idx - 1];
            int gy = gray.data[idx + width] - gray.data[idx - width];

            int mag = abs(gx) + abs(gy);

            if (mag > 100)
                output.data[idx] = 255;
            else
                output.data[idx] = 0;
        }
    }

    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("CPU Time: %.2f ms\n", time_taken);

    // Save output
    imwrite("cpu_edges.png", output);

    printf("Output saved as cpu_edges.png\n");

    return 0;
}
