#!/bin/bash

# Set the build directory
BUILD_DIR=build

# Remove the existing build directory if it exists
if [ -d "$BUILD_DIR" ]; then
    echo "Removing existing build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create a new build directory
echo "Creating build directory..."
mkdir "$BUILD_DIR"

# Navigate into the build directory
cd "$BUILD_DIR"

# Run CMake to configure the project
echo "Configuring the project with CMake..."
cmake ..

# Build the project
echo "Building the project..."
make

# Navigate back to the root directory
cd ..

echo "Build process completed."