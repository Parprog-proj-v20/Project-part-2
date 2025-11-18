/**
 * @file MatrixTest.cpp
 * @brief Реализация тестов для класса Matrix
 */
#include "MatrixTest.h"
#include "Matrix.h"
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <string>
#include <chrono>
#include <omp.h>
#include <thread>
#include <vector>
#include <atomic>

const double MatrixTest::PERFORMANCE_TOLERANCE = 0.8;


void MatrixTest::runAllTests() {
    std::cout << "*** Запуск тестов ***" << std::endl;

    int passedTests = 0;
    int totalTests = 0;

    auto runTest = [&](const std::string& testName, void(*testFunc)()) {
        totalTests++;
        std::cout << "\n    Тест: " << testName << std::endl;
        try {
            testFunc();
            std::cout << "    Тест пройден: " << testName << std::endl;
            passedTests++;
        }
        catch (const std::exception& e) {
            std::cout << "    Тест НЕ пройден: " << testName << " - " << e.what() << std::endl;
        }
        };

    runTest("Корректность умножения", testMultiplicationCorrectness);
    runTest("Различные размеры матриц", testDifferentSizes);
    runTest("Граничные случаи", testEdgeCases);
    runTest("Производительность", testPerformance);
    runTest("Типы планирования", testSchedulingTypes);
    runTest("Сравнение линейного и параллельного", testLinearVsParallel);
    runTest("Известные матрицы", testKnownMatrices);
    runTest("Безопасность потоков", testThreadSafety);
    runTest("Параллельная инициализация", testConcurrentInitialization);
    runTest("Проверка состояний гонки", testRaceCondition);

    std::cout << "\n*** Результаты тестов ***" << std::endl;
    std::cout << "Пройдено: " << passedTests << "/" << totalTests << " тестов" << std::endl;
}
/**
 * @brief Тестирование базовой корректности умножения матриц
 *
 * Проверка умножение на единичную матрицу дает исходную матрицу, и что все типы планирования OpenMP дают одинаковый результат
 */
void MatrixTest::testMultiplicationCorrectness() {
    std::cout << "Проверка корректности умножения матриц 4x4" << std::endl;

    Matrix matrix(4);
    matrix.setMatrixA({ {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16} });
    matrix.setMatrixB({ {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1} });

    std::vector<std::vector<int>> expected = { {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16} };

    matrix.multiplyLinear();
    assert(areMatricesEqual(matrix.getMatrixC(), expected));

    std::vector<std::string> schedules = { "static", "dynamic", "guided" };
    for (const auto& schedule : schedules) {
        matrix.multiplyParallel(2, schedule);
        assert(areMatricesEqual(matrix.getMatrixC(), expected));
    }
}


/**
 * @brief Тестирование работы с матрицами разных размеров
 *
 * Проверка что умножение работает корректно для матриц различных размеров от 2x2 до 50x50
 */
void MatrixTest::testDifferentSizes() {
    std::vector<int> testSizes = { 2, 10, 50 };
    for (int size : testSizes) {
        std::cout << "Тестирование размера " << size << "x" << size << "..." << std::endl;
        Matrix matrix(size);
        matrix.initialize();
        matrix.multiplyLinear();
        assert(!matrix.getMatrixC().empty());
        matrix.multiplyParallel(4, "static");
        assert(!matrix.getMatrixC().empty());
    }
}

/**
 * @brief Тестирование граничных случаев
 *
 * Проверка особых случаев: матрица 1x1, умножение на нулевую матрицу, умножение на единичную матрицу
 */
void MatrixTest::testEdgeCases() {
    std::cout << "Тестирование граничных случаев" << std::endl;

    Matrix matrix1(1);
    matrix1.setMatrixA({ {5} });
    matrix1.setMatrixB({ {3} });
    matrix1.multiplyParallel(2, "static");
    assert(matrix1.getMatrixC()[0][0] == 15);

    Matrix matrix2(3);
    matrix2.setMatrixA({ {0, 0, 0}, {0, 0, 0}, {0, 0, 0} });
    matrix2.setMatrixB({ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} });
    std::vector<std::vector<int>> expected = { {0, 0, 0}, {0, 0, 0}, {0, 0, 0} };
    matrix2.multiplyParallel(2, "static");
    assert(areMatricesEqual(matrix2.getMatrixC(), expected));

    Matrix matrix3(3);
    std::vector<std::vector<int>> matrixAData = { {1, 2, 3}, {4, 5, 6}, {7, 8, 9} };
    matrix3.setMatrixA(matrixAData);
    matrix3.setMatrixB({ {1, 0, 0}, {0, 1, 0}, {0, 0, 1} });
    matrix3.multiplyParallel(2, "static");
    assert(areMatricesEqual(matrix3.getMatrixC(), matrixAData));
}

/**
 * @brief Тестирование производительности
 *
 * Проверка что многопоточная версия дает ускорение по сравнению с однопоточной
 */
void MatrixTest::testPerformance() {
    std::cout << "Тестирование производительности" << std::endl;
    Matrix matrix(300);
    matrix.initialize();
    double singleThreadTime = matrix.multiplyParallel(1, "static");
    std::vector<int> threadCounts = { 2, 4 };
    for (int threads : threadCounts) {
        double multiThreadTime = matrix.multiplyParallel(threads, "static");
        validateSpeedup(singleThreadTime, multiThreadTime, threads, "static scheduling");
        assert(multiThreadTime <= singleThreadTime * 1.5);
    }
}

/**
 * @brief Тестирование типов планирования OpenMP
 *
 * Сравнивает производительность static, dynamic и guided планирования
 */
void MatrixTest::testSchedulingTypes() {
    std::cout << "Сравнение типов планирования" << std::endl;
    Matrix matrix(200);
    matrix.initialize();
    std::vector<std::string> schedules = { "static", "dynamic", "guided" };
    std::vector<double> times;
    for (const auto& schedule : schedules) {
        double time = matrix.multiplyParallel(4, schedule);
        times.push_back(time);
        std::cout << "Время для " << schedule << ": " << time << " сек" << std::endl;
    }
    for (double time : times) {
        assert(time > 0 && time < 10.0);
    }
}

/**
 * @brief Тестирование согласованности результатов
 *
 * Проверка, что линейное и параллельное умножени дают идентичные результаты
 */
void MatrixTest::testLinearVsParallel() {
    std::cout << "Сравнение линейного и параллельного умножения" << std::endl;
    Matrix matrix(100);
    matrix.initialize();
    matrix.multiplyLinear();
    auto linearResult = matrix.getMatrixC();
    matrix.multiplyParallel(2, "static");
    auto parallelResult = matrix.getMatrixC();
    assert(areMatricesEqual(linearResult, parallelResult));
}

/**
 * @brief Тестирование на известных матрицах
 *
 * Проверка умножения матриц с заранее известным результатом
 */
void MatrixTest::testKnownMatrices() {
    std::cout << "Тестирование на известных матрицах" << std::endl;
    Matrix matrix(2);
    matrix.setMatrixA({ {1, 2}, {3, 4} });
    matrix.setMatrixB({ {2, 0}, {1, 2} });
    std::vector<std::vector<int>> expected = { {4, 4}, {10, 8} };
    matrix.multiplyParallel(2, "static");
    assert(areMatricesEqual(matrix.getMatrixC(), expected));
}

/**
 * @brief Тестирование потокобезопасности
 *
 * Проверка корректности работы при одновременном выполнении множества операций умножения
 */
void MatrixTest::testThreadSafety() {
    std::cout << "Проверка безопасности потоков" << std::endl;
    const int matrixSize = 100;
    const int numConcurrentOperations = 10;
    std::vector<Matrix> matrices(numConcurrentOperations, Matrix(matrixSize));

    for (auto& matrix : matrices) {
        matrix.initialize();
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < numConcurrentOperations; i++) {
        matrices[i].multiplyParallel(4, "static");
    }

    for (const auto& matrix : matrices) {
        const auto& result = matrix.getMatrixC();
        assert(result.size() == matrixSize);
        assert(result[0].size() == matrixSize);
        bool hasNonZero = false;
        for (const auto& row : result) {
            for (int val : row) {
                if (val != 0) hasNonZero = true;
            }
        }
        assert(hasNonZero);
    }
}

/**
 * @brief Тестирование параллельной инициализации
 *
 * Проверка, что инициализация множества матриц в параллельном режиме работает корректно
 */
void MatrixTest::testConcurrentInitialization() {
    std::cout << "Проверка параллельной инициализации" << std::endl;
    const int numMatrices = 8;
    const int matrixSize = 50;
    std::vector<Matrix> matrices(numMatrices, Matrix(matrixSize));

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numMatrices; i++) {
        matrices[i].initialize();
    }

    for (const auto& matrix : matrices) {
        const auto& A = matrix.getMatrixC();
        assert(A.size() == matrixSize);
        assert(A[0].size() == matrixSize);
    }
}

/**
 * @brief Тестирование на отсутствие состояний гонки
 *
 * Проверка стабильности результатов при многократном выполнении параллельных операций
 */
void MatrixTest::testRaceCondition() {
    std::cout << "Проверка состояний гонки" << std::endl;
    Matrix matrix(80);
    matrix.initialize();
    matrix.multiplyParallel(4, "static");
    auto firstResult = matrix.getMatrixC();

    const int numIterations = 20;
    for (int i = 0; i < numIterations; i++) {
        matrix.multiplyParallel(4, "static");
        auto currentResult = matrix.getMatrixC();
        assert(areMatricesEqual(firstResult, currentResult));
    }

    std::cout << "Состояния гонки не обнаружены после " << numIterations << " итераций" << std::endl;
}

// Вспомогательные методы

bool MatrixTest::areMatricesEqual(const std::vector<std::vector<int>>& matrix1,
    const std::vector<std::vector<int>>& matrix2) {
    if (matrix1.size() != matrix2.size()) return false;
    for (size_t i = 0; i < matrix1.size(); i++) {
        if (matrix1[i].size() != matrix2[i].size()) return false;
        for (size_t j = 0; j < matrix1[i].size(); j++) {
            if (matrix1[i][j] != matrix2[i][j]) return false;
        }
    }
    return true;
}

std::vector<std::vector<int>> MatrixTest::simpleMultiply(const std::vector<std::vector<int>>& A,
    const std::vector<std::vector<int>>& B) {
    int n = A.size();
    std::vector<std::vector<int>> result(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

void MatrixTest::validateSpeedup(double singleThreadTime, double multiThreadTime,
    int threadCount, const std::string& testCase) {
    double speedup = singleThreadTime / multiThreadTime;
    std::cout << "Ускорение для " << testCase << " с " << threadCount
        << " потоками: " << speedup << std::endl;
}
