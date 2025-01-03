#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// Kernel to check if three points are collinear
__device__ bool areCollinearKernel(float ax, float ay, float bx, float by, float cx, float cy) {
    return (by - ay) * (cx - bx) == (cy - by) * (bx - ax);
}

// Kernel to check if a point is inside a circumcircle
__device__ bool pointInCircleKernel(float px, float py, float* circle) {
    float dx = px - circle[0];
    float dy = py - circle[1];
    float distance = sqrt(dx * dx + dy * dy);
    return distance < circle[2];
}

// Kernel to compute circumcircle of a triangle
__device__ void computeCircumcircleKernel(float ax, float ay, float bx, float by, float cx, float cy, float* circle) {
    float D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    circle[0] = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / D;
    circle[1] = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / D;
    circle[2] = sqrt((circle[0] - ax) * (circle[0] - ax) + (circle[1] - ay) * (circle[1] - ay));
}

// Kernel to check if a triangle is Delaunay
__device__ bool triangleIsDelaunayKernel(float ax, float ay, float bx, float by, float cx, float cy, float* points, int numPoints) {
    float circle[3];
    computeCircumcircleKernel(ax, ay, bx, by, cx, cy, circle);
    for (int i = 0; i < numPoints; ++i) {
        float px = points[2 * i];
        float py = points[2 * i + 1];
        if ((px != ax || py != ay) && (px != bx || py != by) && (px != cx || py != cy)) {
            if (pointInCircleKernel(px, py, circle)) {
                return false;
            }
        }
    }
    return true;
}

// CUDA Kernel to perform triangulation
__global__ void triangulationKernel(float* points, int numPoints, int* triangles, int* numTriangles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes a unique triangle
    int i = idx / (numPoints * numPoints);
    int j = (idx / numPoints) % numPoints;
    int k = idx % numPoints;

    if (i >= numPoints || j >= numPoints || k >= numPoints || j <= i || k <= j) return;

    float ax = points[2 * i];
    float ay = points[2 * i + 1];
    float bx = points[2 * j];
    float by = points[2 * j + 1];
    float cx = points[2 * k];
    float cy = points[2 * k + 1];

    if (!areCollinearKernel(ax, ay, bx, by, cx, cy)) {
        if (triangleIsDelaunayKernel(ax, ay, bx, by, cx, cy, points, numPoints)) {
            int triangleIdx = atomicAdd(numTriangles, 1);
            triangles[3 * triangleIdx] = i;
            triangles[3 * triangleIdx + 1] = j;
            triangles[3 * triangleIdx + 2] = k;
        }
    }
}

// Host function to call the kernel
void triangulationCUDA(float* points, int numPoints, std::vector<std::vector<int>>& triangles) {
    int maxTriangles = numPoints * (numPoints - 1) * (numPoints - 2) / 6;

    // Allocate device memory
    float* d_points;
    int* d_triangles;
    int* d_numTriangles;
    int h_numTriangles = 0;

    cudaMalloc(&d_points, 2 * numPoints * sizeof(float));
    cudaMalloc(&d_triangles, 3 * maxTriangles * sizeof(int));
    cudaMalloc(&d_numTriangles, sizeof(int));

    cudaMemcpy(d_points, points, 2 * numPoints * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numTriangles, &h_numTriangles, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int totalThreads = numPoints * numPoints * numPoints;
    int blockSize = 256;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;

    triangulationKernel << <numBlocks, blockSize >> > (d_points, numPoints, d_triangles, d_numTriangles);

    // Copy results back to host
    cudaMemcpy(&h_numTriangles, d_numTriangles, sizeof(int), cudaMemcpyDeviceToHost);
    int* h_triangles = new int[3 * h_numTriangles];
    cudaMemcpy(h_triangles, d_triangles, 3 * h_numTriangles * sizeof(int), cudaMemcpyDeviceToHost);

    // Store triangles in vector
    triangles.clear();
    for (int i = 0; i < h_numTriangles; ++i) {
        triangles.push_back({ h_triangles[3 * i], h_triangles[3 * i + 1], h_triangles[3 * i + 2] });
    }

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_triangles);
    cudaFree(d_numTriangles);
    delete[] h_triangles;
}
