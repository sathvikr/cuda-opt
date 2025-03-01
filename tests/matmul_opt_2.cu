```cpp
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmulKernelOptimized(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int ph = 0; ph < (N + TILE_SIZE - 1) / TILE_SIZE; ++ph) {
        // Load tiles into shared memory
        int aCol = ph * TILE_SIZE + tx;
        int bRow = ph * TILE_SIZE + ty;

        if (row < N && aCol < N)
            As[ty][tx] = A[row * N + aCol];
        else
            As[ty][tx] = 0.0f;

        if (bRow < N && col < N)
            Bs[ty][tx] = B[bRow * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

void matmul(float* A, float* B, float* C, int N) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (N + blockDim.y - 1) / blockDim.y);
    matmulKernelOptimized<<<gridDim, blockDim>>>(A, B, C, N);
}
```