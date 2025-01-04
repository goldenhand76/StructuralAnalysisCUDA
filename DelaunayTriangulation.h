#pragma once
#ifndef DELAUNAYTRIANGULATION_H
#define DELAUNAYTRIANGULATION_H

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

class Point {
public:
    float x, y;

    Point(float x, float y);
    std::vector<float> pos() const;
    bool isEqual(const Point& other) const;
    std::string pointToStr() const;
};

class Triangle {
public:
    Point a, b, c;

    Triangle() : a({ 0, 0 }), b({ 0, 0 }), c({ 0, 0 }) {}

    Triangle(const Point& a, const Point& b, const Point& c);
    bool isEqual(const Triangle& other) const;
};

class Graph {
private:
    std::vector<float> circumcircle(const Triangle& tri) const;
    bool pointInCircle(const Point& point, const std::vector<float>& circle) const;
    bool areCollinear(const Point& a, const Point& b, const Point& c);
    void partitionPoints(const std::vector<Point>& points, std::vector<std::vector<Point>>& domains, int numDomains);

public:
    std::vector<Point> points;
    std::vector<Triangle> triangles;

    int getNodeIndex(const Point& point) const;
    void addPoint(const Point& point);
    void triangulation(float* apoints, int rank, int size, int smSize);
    void triangulationWithLocalStorage(int rank, int size);
    void triangulationWithTaskPool();
    void triangulationWithDomainDecomposition(int numDomains);
    bool triangleIsDelaunay(const Triangle& triangle) const;
    void printTriangles() const;

};

#endif // DELAUNAYTRIANGULATION_H
