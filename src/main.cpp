#include <iostream>
#include <vector>
#include <iomanip>
#include "Matrix.h"
#ifdef _WIN32 
#include <windows.h>
#endif

/**
* @brief Создание матриц установленных размеров и их перемножение при разном количестве
* потоком и разных типах планировок. Вызывает функции для получения времени линейного
* перемножения и однопоточного перемножения с планировкой static, затем сравнивает время
* выполнения и рассчитывает ускорение.
*/
void run() {
    double time, speedup, linear_speedup;
    std::vector<int> sizes = { 100, 200 };
    std::vector<int> thread_counts = { 2, 4, 8 };
    std::vector<std::string> types = { "static", "dynamic", "guided" };
    
    for (int size : sizes) {
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "\n\tПЕРЕМНОЖЕНИЕ МАТРИЦ: A[" << size << "x" << size << "] * B[" << size << "x" << size << "]\n" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        Matrix m(size);
        m.initialize();

        double linear_time = m.multiplyLinear();
        double single_thread_time = m.multiplyParallel(1, "static");
        std::cout << "\t> ЛИНЕЙНОЕ ПЕРЕМНОЖЕНИЕ, время выполнения: " << linear_time << " сек" << std::endl;
        std::cout << "\t> 1 ПОТОК, время выполнения: " << single_thread_time << " сек" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        for (int threads : thread_counts) {
            std::cout << std::string(60, '=') << std::endl;
            std::cout << "\t> КОЛИЧЕСТВО ПОТОКОВ: " << threads << "; МАТРИЦЫ: " << size << "x" << size << std::endl;
            std::cout << std::string(60, '=') << std::endl;

            for (const auto& schedule : types) {
                std::cout << "\t> ПЛАНИРОВКА: " << schedule << "\n\t> ПОТОКОВ: " << threads << "\n" << std::endl;
                time = m.multiplyParallel(threads, schedule);
                speedup = single_thread_time / time;
                linear_speedup = linear_time / time;

                std::cout << "\n\t> ВРЕМЯ: " << time << " сек\n\tУскорение по отнош. к линейному: " 
                    << linear_speedup << "\n\tУскорение по отнош.к 1 потоку: " << speedup << std::endl;
                std::cout << std::string(60, '-') << std::endl;
            }
        }
    }
}

int main() {
    #ifdef _WIN32
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
    #endif

    run();
    return 0;
}