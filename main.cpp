#include <iostream>
#include "./cuda_kernel.cuh"
#include "omp.h"
#include "mpi.h"
#include <cuda_runtime.h>
#include "./PointGenerator.cuh"

#define NUM_THREADS 5

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

    // Initialize arrays A, B, and C.
    double A[3], B[3], C[3];

    // Populate arrays A and B.
    A[0] = 5; A[1] = 8; A[2] = 3;
    B[0] = 7; B[1] = 6; B[2] = 4;

    // Sum array elements across ( C[0] = A[0] + B[0] ) into array C using CUDA.
    kernel(A, B, C, 3);

    // Print out result.
    std::cout << "C = " << C[0] << ", " << C[1] << ", " << C[2] << std::endl;

    float hostPoints[400];
    float hostFlags[400];
    generateOrderedPointsWithHole(hostPoints, hostFlags, 20, 5);

    for (int i = 0; i < 400; i++) {
        std::cout << "Point = " << hostPoints[i] << std::endl;
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