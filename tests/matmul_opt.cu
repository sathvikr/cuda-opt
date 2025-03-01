To optimize the provided CUDA matrix multiplication kernel, we focus on improving memory access patterns and leveraging shared memory to reduce global memory accesses. Here's the step-by-step explanation and the optimized code:

### Optimizations Applied:
1. **Tiling with Shared Memory**: Load tiles of matrices A and B into shared memory to reuse data and reduce global memory accesses.
2. **Coalesced Memory Access**: Ensure global memory accesses are coalesced by having consecutive threads read consecutive memory locations.
3. **Loop Unrolling**: Unroll inner loops to reduce overhead and improve instruction-level parallelism.
4. **Boundary Checks**: Handle cases where matrix dimensions aren't multiples of the tile size.
5. **Use of `__restrict__`**: Inform the compiler that pointers do not alias, enabling better optimization.

### Optimized Code:
```cpp
#include <cuda_runtime.h>

#define TILE_SIZE 16  // Optimal tile size for shared memory

__global__ void matmulKernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int k = 0; k < N; k += TILE_SIZE) {
        // Load tiles into shared memory with boundary checks
        if (row < N && (k + tx) < N)
            As[ty][tx] = A[row * N + (k + tx)];
        else
            As[ty][tx] = 0.0f;

        if (col < N && (k + ty) < N)
            Bs[ty][tx] = B[(k + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute product of tiles with loop unrolling
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    // Write result to global memory
    if (row < N && col < N)
        C[row * N + col] = sum;
}

// Host function to launch kernel
void matmul(float* A, float* B, float* C, int N) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, 
                 (N + TILE_SIZE - 1) / TILE_SIZE);

    matmulKernel<<<gridDim, blockDim>>>(A, B, C, N);
}
```

### Explanation:
1. **Shared Memory Tiles**: 
   - `As` and `Bs` are shared memory arrays storing tiles of matrices A and B, respectively.
   - Each thread loads one element from A's tile and one from B's tile into shared memory, promoting data reuse.

2. **Coalesced Access**:
   - Threads in a block load contiguous elements from global memory into shared memory, ensuring coalesced accesses.
   - For matrix A, threads load elements row-wise; for B, column-wise loading is avoided by using shared memory effectively.

3. **Loop Unrolling**:
   - The `#pragma unroll` directive unrolls the inner loop, reducing loop overhead and improving performance.

4. **Boundary Handling**:
   - Checks ensure threads don't access out-of-bounds elements, setting them to zero if beyond matrix dimensions.

5. **Grid and Block Configuration**:
   - Blocks are sized as `TILE_SIZE x TILE_SIZE` (16x16), balancing shared memory usage and thread occupancy.

### Additional Considerations:
- **Tile Size**: Experiment with tile sizes (e.g., 32x32) depending on GPU architecture and shared memory limits.
- **Pre-transposing B**: For even better performance, pre-transpose matrix B to allow more efficient access patterns.
- **CUDA Streams**: Use asynchronous memory operations and streams to overlap computation with data transfers for larger matrices.

This optimized kernel leverages shared memory to reduce global memory bandwidth usage and coalesced accesses to maximize memory throughput, significantly improving performance over the naive implementation.