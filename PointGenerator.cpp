#include "DelaunayTriangulation.h"
#include "chrono"
#include <random>

void generateOrderedPointsWithHole(Graph& graph, float squareSize, float holeSize) {

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
}


void generateRandomPointsWithHole(Graph& graph, float squareSize, float holeSize) {

    // Define the boundaries of the hole
    float halfSquare = squareSize / 2;
    float halfHole = holeSize / 2;
    float holeMinX = halfSquare - halfHole;
    float holeMaxX = halfSquare + halfHole;
    float holeMinY = halfSquare - halfHole;
    float holeMaxY = halfSquare + halfHole;

    int numberPoints = squareSize * squareSize;
    std::default_random_engine eng(std::random_device{}());
    std::uniform_int_distribution<int> dist_w(0, squareSize);
    std::uniform_int_distribution<int> dist_h(0, squareSize);

    for (int i = 0; i < numberPoints; ++i) {
        Point point = Point(dist_w(eng), dist_h(eng));
        if (!(point.x >= holeMinX && point.x <= holeMaxX && point.y >= holeMinY && point.y <= holeMaxY)) {
            graph.addPoint(point);
        }
    }
}