#include <cuda_runtime.h>
#include "./PointGenerator.cuh"
#include "DelaunayTriangulation.h"

// CUDA kernel
__global__ void generateOrderedPointsKernel(float* devicePoints, int* pointFlags,
    float squareSize, float holeMinX, float holeMaxX,
    float holeMinY, float holeMaxY, int gridSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx > gridSize || idy > gridSize) return;

    float x = static_cast<float>(idx);
    float y = static_cast<float>(idy);

    // Linear index for the flat array
    int index = idy * (gridSize + 1) + idx;

    // Check if the point is outside the hole boundaries
    if (!(x >= holeMinX && x <= holeMaxX && y >= holeMinY && y <= holeMaxY)) {
        devicePoints[2 * index] = x;
        devicePoints[2 * index + 1] = y;
        pointFlags[index] = 1; // Mark as valid
    }
    else {
        pointFlags[index] = 0; // Mark as invalid
    }
}

void generateOrderedPointsWithHole(float* hostPoints, float* hostFlags, float squareSize, float holeSize) {
    float halfSquare = squareSize / 2;
    float halfHole = holeSize / 2;
    float holeMinX = halfSquare - halfHole;
    float holeMaxX = halfSquare + halfHole;
    float holeMinY = halfSquare - halfHole;
    float holeMaxY = halfSquare + halfHole;

    int gridSize = static_cast<int>(squareSize);
    int numPoints = (gridSize + 1) * (gridSize + 1);

    // Allocate device memory
    float* devicePoints;
    int* deviceFlags;
    cudaMalloc(&devicePoints, 2 * numPoints * sizeof(float));
    cudaMalloc(&deviceFlags, numPoints * sizeof(int));

    // Initialize device memory
    cudaMemset(devicePoints, -1, 2 * numPoints * sizeof(float));
    cudaMemset(deviceFlags, 0, numPoints * sizeof(int));

    // Define CUDA grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSizeDim((gridSize + blockSize.x) / blockSize.x,
        (gridSize + blockSize.y) / blockSize.y);

    // Launch kernel
    generateOrderedPointsKernel << <gridSizeDim, blockSize >> > (
        devicePoints, deviceFlags, squareSize, holeMinX, holeMaxX, holeMinY, holeMaxY, gridSize);

    // Copy results back to host
    cudaMemcpy(hostPoints, devicePoints, 2 * numPoints * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostFlags, deviceFlags, numPoints * sizeof(int), cudaMemcpyDeviceToHost);

    // Add valid points to the graph
    //for (int i = 0; i < numPoints; ++i) {
    //    if (hostFlags[i] == 1) {
    //        float x = hostPoints[2 * i];
    //        float y = hostPoints[2 * i + 1];
    //        graph.addPoint(Point(x, y));
    //    }
    //}

    // Free device memory
    //cudaFree(devicePoints);
    //cudaFree(deviceFlags);
}
