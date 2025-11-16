#pragma once
#include "../Matrix.h"
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <string>

class MatrixTest {
private:
    static const double PERFORMANCE_TOLERANCE;

public:
    static void runAllTests();
    static void testMultiplicationCorrectness();
    static void testDifferentSizes();
    static void testEdgeCases();
    static void testPerformance();
    static void testSchedulingTypes();
    static void testLinearVsParallel();
    static void testKnownMatrices();

private:
    static bool areMatricesEqual(const std::vector<std::vector<int>>& matrix1, 
                                const std::vector<std::vector<int>>& matrix2);
    static std::vector<std::vector<int>> simpleMultiply(const std::vector<std::vector<int>>& A, 
                                                       const std::vector<std::vector<int>>& B);
    static void validateSpeedup(double singleThreadTime, double multiThreadTime, 
                               int threadCount, const std::string& testCase);
};
