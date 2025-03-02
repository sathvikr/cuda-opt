# CUDA Optimizer

A tool for optimizing CUDA kernels and analyzing GPU performance.

## Overview

CUDA Optimizer is a specialized tool designed to help developers optimize their CUDA kernels and improve GPU performance. It provides analysis, suggestions, and automated optimizations for CUDA code.

## Features

- Automatic kernel analysis
- Performance bottleneck detection
- Memory access pattern optimization
- Thread block size optimization
- Shared memory usage recommendations
- Warp efficiency analysis

## Prerequisites

- CUDA Toolkit (11.0 or higher)
- NVIDIA GPU with compute capability 6.0 or higher
- Python 3.8 or higher
- pip package manager

## Installation

Simply install using pip in editable mode:

```bash
git clone https://github.com/yourusername/cuda-opt.git
cd cuda-opt
pip install -e .
```

That's it! Now you can use the `cuda-opt` command anywhere in your terminal.

## Usage

1. Basic kernel analysis:
```bash
cuda-opt analyze <kernel_file.cu>
```

2. Optimization suggestions:
```bash
cuda-opt optimize <kernel_file.cu>
```

3. Performance profiling:
```bash
cuda-opt profile <kernel_file.cu>
```

## Example

Here's an example of optimizing a matrix multiplication kernel:

```cuda
// Before optimization
__global__ void matrixMul(float* A, float* B, float* C, int N) {
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
```

Running the optimizer:
```bash
$ cuda-opt optimize matrix_mul.cu

📊 Performance Analysis
------------------------
✨ Current Performance: 127 GFLOPS
🔍 Main Bottlenecks:
   - Non-coalesced memory access
   - Low occupancy (32%)
   - Memory bank conflicts

🛠️ Suggested Optimizations:
   1. Use shared memory tiling
   2. Increase thread block size to 16x16
   3. Add memory padding

⚡ Estimated Improvement: 3.8x speedup

🔄 Applying optimizations...
```

After applying the suggested optimizations:

```cuda
// After optimization
__global__ void matrixMul(float* A, float* B, float* C, int N) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    
    float sum = 0.0f;
    
    for (int m = 0; m < N/16; m++) {
        As[ty][tx] = A[row * N + (m * 16 + tx)];
        Bs[ty][tx] = B[(m * 16 + ty) * N + col];
        __syncthreads();
        
        for (int k = 0; k < 16; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

📈 Results:
```
Performance Improvement:
┌────────────────┬────────────┬─────────────┐
│     Metric     │   Before   │    After    │
├────────────────┼────────────┼─────────────┤
│ GFLOPS         │    127     │     486     │
│ Occupancy      │     32%    │      78%    │
│ Memory Bandwidth│  142 GB/s  │   389 GB/s  │
│ Bank Conflicts  │    High    │     Low     │
└────────────────┴────────────┴─────────────┘
```

## Configuration

The tool can be configured using a `config.json` file in the project root directory. Example configuration:

```json
{
  "analysis_level": "detailed",
  "optimization_targets": ["memory", "occupancy", "arithmetic"],
  "max_threads_per_block": 1024
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and support, please open an issue in the GitHub repository.

