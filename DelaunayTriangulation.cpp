#include "DelaunayTriangulation.h"
#include "./DelaunayTriangulation_kernel.cuh"

#include <omp.h>  // Include OpenMP header
#include <algorithm>
#include <cmath>
#include <vector>
#include <mpi.h>

// Point class methods (same as before)
Point::Point(float x, float y) : x(x), y(y) {}

std::vector<float> Point::pos() const {
    return { x, y };
}

bool Point::isEqual(const Point& other) const {
    return (x == other.x && y == other.y);
}

std::string Point::pointToStr() const {
    return "(" + std::to_string(x) + ", " + std::to_string(y) + ")";
}

Triangle::Triangle(const Point& a, const Point& b, const Point& c) : a(a), b(b), c(c) {}

bool Triangle::isEqual(const Triangle& other) const {
    return (a.isEqual(other.a) || a.isEqual(other.b) || a.isEqual(other.c)) &&
        (b.isEqual(other.a) || b.isEqual(other.b) || b.isEqual(other.c)) &&
        (c.isEqual(other.a) || c.isEqual(other.b) || c.isEqual(other.c));
}

std::vector<float> Graph::circumcircle(const Triangle& tri) const {
    float D = (tri.a.x - tri.c.x) * (tri.b.y - tri.c.y) - (tri.b.x - tri.c.x) * (tri.a.y - tri.c.y);

    float centerX = (((tri.a.x - tri.c.x) * (tri.a.x + tri.c.x) + (tri.a.y - tri.c.y) * (tri.a.y + tri.c.y)) / 2 * (tri.b.y - tri.c.y) -
        ((tri.b.x - tri.c.x) * (tri.b.x + tri.c.x) + (tri.b.y - tri.c.y) * (tri.b.y + tri.c.y)) / 2 * (tri.a.y - tri.c.y)) / D;

    float centerY = (((tri.b.x - tri.c.x) * (tri.b.x + tri.c.x) + (tri.b.y - tri.c.y) * (tri.b.y + tri.c.y)) / 2 * (tri.a.x - tri.c.x) -
        ((tri.a.x - tri.c.x) * (tri.a.x + tri.c.x) + (tri.a.y - tri.c.y) * (tri.a.y + tri.c.y)) / 2 * (tri.b.x - tri.c.x)) / D;

    float radius = std::sqrt(std::pow(tri.c.x - centerX, 2) + std::pow(tri.c.y - centerY, 2));

    return { centerX, centerY, radius };
}

bool Graph::pointInCircle(const Point& point, const std::vector<float>& circle) const {
    float d = std::sqrt(std::pow(point.x - circle[0], 2) + std::pow(point.y - circle[1], 2));
    return (d < circle[2]);
}

void Graph::addPoint(const Point& point) {
    // Check if point is unique before adding
    for (const auto& p : points) {
        if (p.isEqual(point)) {
            return;  // Equivalent point already exists
        }
    }
    points.push_back(point);
}


void Graph::triangulation(float* apoints, int rank, int size) {
    std::vector<Triangle> localTriangles;

    // Split work dynamically among processes (MPI)
    size_t numPoints = points.size();
    size_t pointsPerRank = numPoints / size;
    size_t remainder = numPoints % size;
    size_t start = rank * pointsPerRank;
    size_t end = start + pointsPerRank + (rank < remainder ? 1 : 0);

    // CUDA-based triangulation (assumed to be already parallelized on the GPU)
    std::vector<std::vector<int>> trianglesCUDA;
    triangulationCUDA(apoints, numPoints, trianglesCUDA);

    std::cout << "#Triangles generated with GPU: " << trianglesCUDA.size() << "\n";
    printf("Rank : %d ---------- Processing points from id : %d to id : %d\n", rank, start, end);

    // Parallelized local computation of triangles using OpenMP
#pragma omp parallel
    {
        // Thread-local storage to minimize contention
        std::vector<Triangle> threadLocalTriangles;

        // Distribute outermost loop iterations across threads
#pragma omp for schedule(dynamic)
        for (size_t i = start; i < end; ++i) {
            for (size_t j = i + 1; j < numPoints; ++j) {
                for (size_t k = j + 1; k < numPoints; ++k) {
                    if (!areCollinear(points[i], points[j], points[k])) {
                        Triangle tri(points[i], points[j], points[k]);
                        if (triangleIsDelaunay(tri)) {
                            threadLocalTriangles.push_back(tri);
                        }
                    }
                }
            }
        }

        // Merge thread-local results into shared localTriangles
#pragma omp critical
        localTriangles.insert(localTriangles.end(), threadLocalTriangles.begin(), threadLocalTriangles.end());
    }

    // Serialize local triangles for communication (MPI)
    size_t localCount = localTriangles.size();
    std::vector<double> serializedLocalData(localCount * 6);  // 3 points * 2 coordinates each
    for (size_t i = 0; i < localCount; ++i) {
        serializedLocalData[i * 6 + 0] = localTriangles[i].a.x;
        serializedLocalData[i * 6 + 1] = localTriangles[i].a.y;
        serializedLocalData[i * 6 + 2] = localTriangles[i].b.x;
        serializedLocalData[i * 6 + 3] = localTriangles[i].b.y;
        serializedLocalData[i * 6 + 4] = localTriangles[i].c.x;
        serializedLocalData[i * 6 + 5] = localTriangles[i].c.y;
    }

    // Gather all local data sizes at root using MPI
    std::vector<int> recvCounts(size), displs(size);
    int localDataSize = localCount * 6;
    MPI_Gather(&localDataSize, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate offsets (displacements)
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i - 1] + recvCounts[i - 1];
        }
    }

    // Allocate receive buffer at root
    std::vector<double> allData;
    if (rank == 0) {
        int totalSize = 0;
        for (int i = 0; i < size; ++i) totalSize += recvCounts[i];
        allData.resize(totalSize);
    }

    // Gather all triangles at root using MPI
    MPI_Gatherv(serializedLocalData.data(), localDataSize, MPI_DOUBLE,
        allData.data(), recvCounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Deserialize triangles at root (MPI root process)
    if (rank == 0) {
        triangles.clear();
        for (size_t i = 0; i < allData.size(); i += 6) {
            Point p1(allData[i], allData[i + 1]);
            Point p2(allData[i + 2], allData[i + 3]);
            Point p3(allData[i + 4], allData[i + 5]);
            triangles.emplace_back(p1, p2, p3);
        }
    }
}


void Graph::triangulationWithLocalStorage(int rank, int size) {
    triangles.clear();
    size_t numPoints = points.size();
    size_t pointsPerRank = numPoints / size;
    size_t remainder = numPoints % size;
    size_t start = rank * pointsPerRank;
    size_t end = start + pointsPerRank + (rank < remainder ? 1 : 0);
    std::vector<std::vector<Triangle>> threadTriangles(omp_get_max_threads());

    for (size_t i = start; i < end; ++i) {
#pragma omp parallel
        {
            int tid = omp_get_thread_num();

#pragma omp for nowait // Parallelize loop without critical section
            for (int j = i + 1; j < numPoints; ++j) {
                for (size_t k = j + 1; k < numPoints; ++k) {
                    if (!areCollinear(points[i], points[j], points[k])) {
                        Triangle newTri(points[i], points[j], points[k]);
                        if (triangleIsDelaunay(newTri)) {
                            threadTriangles[tid].push_back(newTri); // Use thread-local storage
                        }
                    }
                }
            }
        }
    }
    for (const auto& localTriangles : threadTriangles) {
        triangles.insert(triangles.end(), localTriangles.begin(), localTriangles.end());
    }

    std::vector<Triangle> localTriangles = triangles;

    // Serialize local triangles for communication
    size_t localCount = localTriangles.size();
    std::vector<double> serializedLocalData(localCount * 6);  // 3 points * 2 coordinates each
    for (size_t i = 0; i < localCount; ++i) {
        serializedLocalData[i * 6 + 0] = localTriangles[i].a.x;
        serializedLocalData[i * 6 + 1] = localTriangles[i].a.y;
        serializedLocalData[i * 6 + 2] = localTriangles[i].b.x;
        serializedLocalData[i * 6 + 3] = localTriangles[i].b.y;
        serializedLocalData[i * 6 + 4] = localTriangles[i].c.x;
        serializedLocalData[i * 6 + 5] = localTriangles[i].c.y;
    }

    // Gather all local data sizes at root
    std::vector<int> recvCounts(size), displs(size);
    int localDataSize = localCount * 6;
    MPI_Gather(&localDataSize, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate offsets (displacements)
    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i - 1] + recvCounts[i - 1];
        }
    }

    // Allocate receive buffer at root
    std::vector<double> allData;
    if (rank == 0) {
        int totalSize = 0;
        for (int i = 0; i < size; ++i) totalSize += recvCounts[i];
        allData.resize(totalSize);
    }

    // Gather all triangles at root
    MPI_Gatherv(serializedLocalData.data(), localDataSize, MPI_DOUBLE,
        allData.data(), recvCounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Deserialize triangles at root
    if (rank == 0) {
        triangles.clear();
        for (size_t i = 0; i < allData.size(); i += 6) {
            Point p1(allData[i], allData[i + 1]);
            Point p2(allData[i + 2], allData[i + 3]);
            Point p3(allData[i + 4], allData[i + 5]);
            triangles.emplace_back(p1, p2, p3);
        }
    }

}



void Graph::triangulationWithTaskPool() {
    // Clear previous triangles
    triangles.clear();

    size_t numPoints = points.size();
    std::vector<Triangle> tempTriangles;

#pragma omp parallel
    {
#pragma omp single
        {
            for (size_t i = 0; i < numPoints; ++i) {
                for (size_t j = i + 1; j < numPoints; ++j) {
                    for (size_t k = j + 1; k < numPoints; ++k) {
#pragma omp task firstprivate(i, j, k)
                        {
                            if (!areCollinear(points[i], points[j], points[k])) {
                                Triangle newTri(points[i], points[j], points[k]);
                                if (triangleIsDelaunay(newTri)) {
#pragma omp critical
                                    tempTriangles.push_back(newTri);  // Avoid race condition
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Assign the calculated triangles to the main triangles vector
    triangles = tempTriangles;
}



/////////////////////////////////////////////////////////////////////////////////


void Graph::triangulationWithDomainDecomposition(int numDomains) {
    // Enable nested parallelism
    omp_set_nested(1);

    std::vector<std::vector<Point>> domains(numDomains);
    partitionPoints(points, domains, numDomains);

    std::vector<std::vector<Triangle>> domainTriangles(numDomains);

    // Perform triangulation in each domain in parallel
#pragma omp parallel for shared(domainTriangles)
    for (int d = 0; d < numDomains; ++d) {
        std::vector<Triangle> tempTriangles;
        size_t numDomainPoints = domains[d].size();

#pragma omp parallel for shared(tempTriangles)
        for (size_t i = 0; i < numDomainPoints; ++i) {
            for (size_t j = i + 1; j < numDomainPoints; ++j) {
                for (size_t k = j + 1; k < numDomainPoints; ++k) {
                    if (!areCollinear(domains[d][i], domains[d][j], domains[d][k])) {
                        Triangle newTri(domains[d][i], domains[d][j], domains[d][k]);
                        if (triangleIsDelaunay(newTri)) {
#pragma omp critical
                            tempTriangles.push_back(newTri);
                        }
                    }
                }
            }
        }
        domainTriangles[d] = tempTriangles;
    }

    // Merge all triangles from each domain
    for (const auto& triSet : domainTriangles) {
        triangles.insert(triangles.end(), triSet.begin(), triSet.end());
    }
}


bool Graph::triangleIsDelaunay(const Triangle& triangle) const {
    std::vector<float> cc = circumcircle(triangle);
    for (const auto& p : points) {
        if (!p.isEqual(triangle.a) && !p.isEqual(triangle.b) && !p.isEqual(triangle.c)) {
            if (pointInCircle(p, cc)) {
                return false;
            }
        }
    }
    return true;
}


void Graph::printTriangles() const {
    for (const auto& tri : triangles) {
        std::cout << "[" << tri.a.pointToStr() << ", " << tri.b.pointToStr() << ", " << tri.c.pointToStr() << "]," << std::endl;
    }
}


bool Graph::areCollinear(const Point& a, const Point& b, const Point& c) {
    return (b.y - a.y) * (c.x - b.x) == (c.y - b.y) * (b.x - a.x);
}


int Graph::getNodeIndex(const Point& point) const {
    for (size_t i = 0; i < points.size(); ++i) {
        if (points[i].isEqual(point)) {
            return i;
        }
    }
    return -1;
}


void Graph::partitionPoints(const std::vector<Point>& points, std::vector<std::vector<Point>>& domains, int numDomains) {
    if (numDomains <= 0 || points.empty()) return;

    // Determine the grid dimensions for the spatial partitioning
    int gridSize = static_cast<int>(std::sqrt(numDomains));
    if (gridSize * gridSize < numDomains) {
        gridSize += 1;
    }

    float minX = points[0].x, maxX = points[0].x;
    float minY = points[0].y, maxY = points[0].y;

    for (const auto& point : points) {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }

    // Calculate the width and height of each cell in the grid
    float cellWidth = (maxX - minX) / gridSize;
    float cellHeight = (maxY - minY) / gridSize;

    // Resize domains to have numDomains empty vectors
    domains.resize(numDomains);

    // Assign each point to a domain based on its grid cell
    for (const auto& point : points) {
        int xIdx = std::min(static_cast<int>((point.x - minX) / cellWidth), gridSize - 1);
        int yIdx = std::min(static_cast<int>((point.y - minY) / cellHeight), gridSize - 1);

        int domainIdx = yIdx * gridSize + xIdx;
        domains[domainIdx].push_back(point);
    }
}