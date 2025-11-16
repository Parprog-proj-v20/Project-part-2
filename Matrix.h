#pragma once
#include <vector>
#include <string>

/**
* @brief Класс матриц. Инициализирует три квадратные матрицы A, B, C типа int размером n*n.
* Перемножает A и B линейно и с использованием распараллеливания.
*/
class Matrix {
private:
    std::vector<std::vector<int>> A, B, C;
    int n;

public:
    /**
    * @brief Конструктор матриц заданных размеров. Выделяет память под них.
    * Результирующая матрица C инициализируется как нулевая.
    *
    * @param n - количество строк и столбцов каждой матрицы
    */
    Matrix(int n);

    /**
    * @brief Заполнение матриц A и B случайными числами в диапазоне [-100, 100].
    */
    void initialize();

    /**
    * Линейное умножение матриц (без распараллеливания).
    *
    * @return время выполнения в секундах
    */
    double multiplyLinear();

    /**
    * @brief Параллельное умножение матриц с использованием OpenMP.
    *
    * @param num_threads - количество потоков
    * @param type - тип планирования для OpenMP: static, dynamic или guided
    * @return время выполнения в секундах
    */
    double multiplyParallel(int num_threads, const std::string& type);

    // Методы для тестирования
    void setMatrixA(const std::vector<std::vector<int>>& newA); // Установка матрицы A
    void setMatrixB(const std::vector<std::vector<int>>& newB); // Установка матрицы В
    const std::vector<std::vector<int>>& getMatrixC() const; // Получение результата C
    const std::vector<std::vector<int>>& getMatrixA() const; // Получение результата А
    const std::vector<std::vector<int>>& getMatrixB() const; // Получение результата В
};