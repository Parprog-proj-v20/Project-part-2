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
    * @brief Сеттер первой матрицы A. Копирует матрицу newA во внутреннюю переменную класса.
    *
    * @param newA - матрица, которая будет скопирована в A 
    */
    void setMatrixA(const std::vector<std::vector<int>>& newA); 
    /**
    * @brief Сеттер второй матрицы B. Копирует матрицу newB во внутреннюю переменную класса.
    *
    * @param newB - матрица, которая будет скопирована в B
    */
    void setMatrixB(const std::vector<std::vector<int>>& newB);
    /**
    * @brief Геттер первой матрицы A.
    *
    * @return константная ссылка на матрицу A
    */
    const std::vector<std::vector<int>>& getMatrixA() const; 
    /**
    * @brief Геттер второй матрицы B.
    *
    * @return константная ссылка на матрицу B
    */
    const std::vector<std::vector<int>>& getMatrixB() const; 
    /**
    * @brief Геттер результирующей матрицы C.
    *
    * @return константная ссылка на матрицу C
    */
    const std::vector<std::vector<int>>& getMatrixC() const;

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
};
