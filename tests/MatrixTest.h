/**
 * @file MatrixTest.h
 * @brief Заголовочный файл класса тестирования Matrix
 */

#pragma once
#include "Matrix.h"
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <string>

 /**
  * @class MatrixTest
  * @brief Класс для модульного тестирования функциональности Matrix
  *
  * Содержит comprehensive набор тестов для проверки корректности,
  * производительности и потокобезопасности операций с матрицами
  */
class MatrixTest {
private:
    static const double PERFORMANCE_TOLERANCE; ///< Допуск для проверки ускорения

public:
    /// @brief Запуск всех тестов
    static void runAllTests();

    /// @brief Тест корректности алгоритма умножения
    static void testMultiplicationCorrectness();

    /// @brief Тест работы с матрицами разных размеров
    static void testDifferentSizes();

    /// @brief Тест граничных случаев (нулевые, единичные матрицы)
    static void testEdgeCases();

    /// @brief Тест производительности и ускорения
    static void testPerformance();

    /// @brief Тест разных типов планирования OpenMP
    static void testSchedulingTypes();

    /// @brief Тест согласованности линейного и параллельного умножения
    static void testLinearVsParallel();

    /// @brief Тест на известных матрицах с предсказуемым результатом
    static void testKnownMatrices();

    /// @brief Тест безопасности работы с множеством потоков
    static void testThreadSafety();

    /// @brief Тест параллельной инициализации матриц
    static void testConcurrentInitialization();

    /// @brief Тест на отсутствие состояний гонки
    static void testRaceCondition();

private:
    /**
     * @brief Сравнение двух матриц на равенство
     * @param matrix1 Первая матрица
     * @param matrix2 Вторая матрица
     * @return true если матрицы идентичны, иначе false
     */
    static bool areMatricesEqual(const std::vector<std::vector<int>>& matrix1,
        const std::vector<std::vector<int>>& matrix2);

    /**
     * @brief Эталонная реализация умножения матриц для проверки
     * @param A Первая матрица
     * @param B Вторая матрица
     * @return Результат умножения A×B
     */
    static std::vector<std::vector<int>> simpleMultiply(const std::vector<std::vector<int>>& A,
        const std::vector<std::vector<int>>& B);

    /**
     * @brief Валидация ускорения многопоточной версии
     * @param singleThreadTime Время однопоточного выполнения
     * @param multiThreadTime Время многопоточного выполнения
     * @param threadCount Количество потоков
     * @param testCase Название тестового случая
     */
    static void validateSpeedup(double singleThreadTime, double multiThreadTime,
        int threadCount, const std::string& testCase);
};
