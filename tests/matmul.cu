#include <cuda_runtime.h>

// Naive matrix multiplication kernel
__global__ void matmulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Host function to launch kernel
void matmul(float* A, float* B, float* C, int N) {
    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (N + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    matmulKernel<<<gridDim, blockDim>>>(A, B, C, N);
}
