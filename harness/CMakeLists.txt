
cmake_minimum_required(VERSION 3.10)

# Set the project name and enable CUDA
project(HarnessProject LANGUAGES CXX CUDA)

# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add the executable
add_executable(harness_exe
    harness.cu
    kernels/simple_kernel.cu
)

# Specify include directories
target_include_directories(harness_exe PRIVATE harness/kernels)

# Set properties for CUDA compilation
set_target_properties(harness_exe PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optionally, set the architecture for CUDA
# Uncomment and set your specific architecture if needed
# set(CMAKE_CUDA_ARCHITECTURES 75)
