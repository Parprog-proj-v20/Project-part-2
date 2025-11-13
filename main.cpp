#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include <iomanip>

/**
* 
*/
class Matrix {
private:
    std::vector<std::vector<int>> A, B, C;
    int n; 

public:
    /**
    * Конструктор матриц заданных размеров. Результирующая матрица C
    * инициализируется как нулевая
    * 
    * @param n - количество строк и столбцов каждой матрицы
    */
    Matrix(int n) : n(n) {
        A.resize(n, std::vector<int>(n));
        B.resize(n, std::vector<int>(n));
        C.resize(n, std::vector<int>(n, 0.0));
    }

    /**
    * Заполнение матриц A и B случайными числами
    */
    void initialize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(-100, 100);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = dis(gen);
                B[i][j] = dis(gen);
            }
        }
    }

    /**
    * Обычное линейное умножение
    * 
    * @return время выполнения
    */
    double multiplyLinear() {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        return diff.count();
    }

    /**
    * Параллельное умножение
    * 
    * @return время выполнения
    */
    double multiplyParallel(int num_threads, const std::string& type) {
        omp_set_num_threads(num_threads);

        auto start = std::chrono::high_resolution_clock::now();

        // Статическая планировка
        if (type == "static") {
            #pragma omp parallel for schedule(static) collapse(2)
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < n; k++) {
                        sum += A[i][k] * B[k][j];
                    }
                    C[i][j] = sum;
                }
            }
        }

        // Динамическая планировка
        else if (type == "dynamic") {
            #pragma omp parallel for schedule(dynamic) collapse(2)
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < n; k++) {
                        sum += A[i][k] * B[k][j];
                    }
                    C[i][j] = sum;
                }
            }
        }

        // Управляемая планировка
        else if (type == "guided") {
            #pragma omp parallel for schedule(guided) collapse(2)
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < n; k++) {
                        sum += A[i][k] * B[k][j];
                    }
                    C[i][j] = sum;
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        return duration.count();
    }
};

/**
* 
*/
void run() {
    int n = 200;
    double time, speedup, linear_speedup;

    std::vector<int> thread_counts = { 1, 2, 4, 8 };
    std::vector<std::string> types = { "static", "dynamic", "guided" };

    std::cout << "Перемножение матриц размером: " << n << "x" << n << " * " << n << "x" << n << "\n" << std::endl;

    Matrix m(n);
    m.initialize();

    double linear_time = m.multiplyLinear();
    std::cout << "Время выполнения с линейным перемножением: " << linear_time << " сек" << std::endl;

    double base_time = m.multiplyParallel(1, "static");
    std::cout << "Время выполнения с 1 потоком: " << base_time << " сек\n" << std::endl;

    for (int threads : thread_counts) {
        for (const auto& schedule : types) {
            time = m.multiplyParallel(threads, schedule);
            speedup = base_time / time;
            linear_speedup = linear_time / time;

            std::cout << "Кол-во потоков: " << threads << ", планировка: " << schedule << ", время: " << time
                << "сек\n\tускорение по отнош. к 1 потоку: " << speedup << "\n\tускорение по отнош. к линейному: " << linear_speedup << std::endl;
        }
        std::cout << std::endl;
    }
}

int main() {
    setlocale(LC_ALL, "Russian");
    run();
    return 0;
}
