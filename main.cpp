#include <iostream>
#include <cuda_runtime.h>
#include "omp.h"
#include "mpi.h"
#include "chrono"

#include "DelaunayTriangulation.h"

#define NUM_THREADS 12


float* generatePointsWithHole(Graph& graph, float squareSize, float holeSize) {
    // Define the boundaries of the hole
    float halfSquare = squareSize / 2;
    float halfHole = holeSize / 2;
    float holeMinX = halfSquare - halfHole;
    float holeMaxX = halfSquare + halfHole;
    float holeMinY = halfSquare - halfHole;
    float holeMaxY = halfSquare + halfHole;

    for (float i = 0; i <= squareSize; ++i) {
        for (float j = 0; j <= squareSize; ++j) {
            // Check if the point is outside the hole boundaries
            if (!(i >= holeMinX && i <= holeMaxX && j >= holeMinY && j <= holeMaxY)) {
                graph.addPoint(Point(i, j));
            }
        }
    }

    float* points = new float[graph.points.size() * 2]; // Each point has x and y

    for (int i = 0; i < graph.points.size(); i++) {
        points[i * 2] = graph.points[i].x;
        points[i * 2 + 1] = graph.points[i].y;
    }

    return points;
}

void printMPIProperties(int size);
void printDeviceProperties(int deviceCount);

int main(int argc, char** argv)
{
    omp_set_num_threads(NUM_THREADS);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printMPIProperties(size);
    }

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "\nNumber of CUDA devices: " << deviceCount << "\n" << std::endl;
    printDeviceProperties(deviceCount);
    int smSize = 24;

    auto start = std::chrono::high_resolution_clock::now(); // Start timing

    Graph graph;
    std::vector<Point> points;

    float* apoints = generatePointsWithHole(graph, 20, 5);
    std::cout << "\nNumber of generated points: " << graph.points.size() << "\n" << std::endl;

    graph.triangulation(apoints, rank, size, smSize);
    std::cout << "#Triangles generated with ALL CPUs: " << graph.triangles.size() << "\n";

    auto phase1_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> phase1_duration = phase1_end - start;
    std::cout << "Time taken for Phase 1 (Generating triangles): " << phase1_duration.count() << " seconds." << std::endl;

    if (rank == 0) {
        //graph.printTriangles();
    }

    MPI_Finalize();
}

void printMPIProperties(int size) {
    std::cout << "MPI Environment Details:" << std::endl;
    std::cout << "  Number of processes: " << size << std::endl;
    std::cout << "  Processor name: ";

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    std::cout << processor_name << std::endl;

    // Display version information
    int version, subversion;
    MPI_Get_version(&version, &subversion);
    std::cout << "  MPI Version: " << version << "." << subversion << std::endl;
}

void printDeviceProperties(int deviceCount) {

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        std::cout << "Device " << device << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Registers per block: " << prop.regsPerBlock << std::endl;
        std::cout << "  Warp size: " << prop.warpSize << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads dimension (x, y, z): ("
                  << prop.maxThreadsDim[0] << ", "
                  << prop.maxThreadsDim[1] << ", "
                  << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max grid size (x, y, z): ("
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Clock rate: " << prop.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Multi-processor count: " << prop.multiProcessorCount << std::endl;
        std::cout << "  L2 cache size: " << prop.l2CacheSize / 1024 << " KB" << std::endl;
        std::cout << "  Memory clock rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory bus width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Peak memory bandwidth: "
                  << (2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8)) / 1.0e6
                  << " GB/s" << std::endl;
        std::cout << std::endl;
    }
}