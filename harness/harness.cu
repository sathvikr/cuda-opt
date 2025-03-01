#include <iostream>
#include <cuda_runtime.h>
#include "./kernels/simple_kernel.cuh"

int main() {
    int N = 1024; // Example size, can be adjusted
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize host matrices
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (N + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    cudaEvent_t start, stop;
    float milliseconds = 0;

    // Create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Launch the kernel
    kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Record the stop event
    cudaEventRecord(stop);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate the elapsed time in milliseconds
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print the elapsed time
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result (optional)
    // for (int i = 0; i < N * N; ++i) {
    //     std::cout << h_C[i] << " ";
    //     if ((i + 1) % N == 0) std::cout << std::endl;
    // }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
