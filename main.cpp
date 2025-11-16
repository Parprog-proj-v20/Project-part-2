#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include <iomanip>

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
    Matrix(int n) : n(n) {
        A.resize(n, std::vector<int>(n));
        B.resize(n, std::vector<int>(n));
        C.resize(n, std::vector<int>(n, 0.0));
    }

    /**
    * @brief Заполнение матриц A и B случайными числами в диапазоне [-100, 100].
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
    * Линейное умножение матриц (без распараллеливания). 
    * 
    * @return время выполнения в секундах
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
    * @brief Параллельное умножение матриц с использованием OpenMP.
    * 
    * @param num_threads - количество потоков 
    * @param type - тип планирования для OpenMP: static, dynamic или guided
    * @return время выполнения в секундах
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
* @brief Создание матриц установленных размеров и их перемножение при разном количестве
* потоком и разных типах планировок. Вызывает функции для получения времени линейного
* перемножения и однопоточного перемножения с планировкой static, затем сравнивает время 
* выполнения и рассчитывает ускорение. 
*/
void run() {
    double time, speedup, linear_speedup;
    std::vector<size_t> sizes = { 400, 500, 600, 800, 1000 };
    std::vector<int> thread_counts = { 1, 2, 4, 8 };
    std::vector<std::string> types = { "static", "dynamic", "guided" };
    std::cout << std::string(60, '-') << std::endl;

    for (int size : sizes) {
        std::cout << "\n\tПЕРЕМНОЖЕНИЕ МАТРИЦ: A[" << size << "x" << size << "] * B[" << size << "x" << size << "]\n" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        Matrix m(size);
        m.initialize();

        double linear_time = m.multiplyLinear();
        double single_thread_time = m.multiplyParallel(1, "static");
        std::cout << "> ЛИНЕЙНОЕ ПЕРЕМНОЖЕНИЕ, время выполнения: " << linear_time << " сек" << std::endl;
        std::cout << "> 1 ПОТОК, время выполнения: " << single_thread_time << " сек" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        for (int threads : thread_counts) {
            std::cout << "> КОЛИЧЕСТВО ПОТОКОВ: " << threads << std::endl;

            for (const auto& schedule : types) {
                time = m.multiplyParallel(threads, schedule);
                speedup = single_thread_time / time;
                linear_speedup = linear_time / time;

                std::cout << "\n> ПЛАНИРОВКА: " << schedule << "\n> ВРЕМЯ: " << time
                    << " сек\n\tУскорение по отнош. к линейному: " << linear_speedup << "\n\tУскорение по отнош.к 1 потоку: " << speedup << std::endl;
            }
            std::cout << std::string(60, '-') << std::endl;
        }
        std::cout << "\n" << std::string(60, '-') << std::endl;
    }
}

int main() {
    setlocale(LC_ALL, "Russian");
    run();
    return 0;
}
