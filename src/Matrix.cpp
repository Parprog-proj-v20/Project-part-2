#include "Matrix.h"
#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>

Matrix::Matrix(int n) : n(n) {
    A.resize(n, std::vector<int>(n));
    B.resize(n, std::vector<int>(n));
    C.resize(n, std::vector<int>(n, 0));
}

void Matrix::setMatrixA(const std::vector<std::vector<int>>& newA) {
    A = newA;
}
void Matrix::setMatrixB(const std::vector<std::vector<int>>& newB) {
    B = newB;
}

const std::vector<std::vector<int>>& Matrix::getMatrixA() const {
    return A;
}
const std::vector<std::vector<int>>& Matrix::getMatrixB() const {
    return B;
}
const std::vector<std::vector<int>>& Matrix::getMatrixC() const {
    return C;
}

void Matrix::initialize() {
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

double Matrix::multiplyLinear() {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int sum = 0;
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

double Matrix::multiplyParallel(int num_threads, const std::string& type) {
    omp_set_num_threads(num_threads);
    auto start = std::chrono::high_resolution_clock::now();
    int tid = 0;
    double thread_start_time = 0;
    double thread_time = 0;

    #pragma omp parallel private(tid, thread_start_time, thread_time)
    {
        tid = omp_get_thread_num();
        thread_start_time = omp_get_wtime();
        std::cout << "[Поток " << tid << " запущен]\n";

        // Статическая планировка
        if (type == "static") {
            #pragma omp parallel for schedule(static) collapse(2)
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int sum = 0;
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
                    int sum = 0;
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
                    int sum = 0;
                    for (int k = 0; k < n; k++) {
                        sum += A[i][k] * B[k][j];
                    }
                    C[i][j] = sum;
                }
            }
        }
        
        thread_time = omp_get_wtime() - thread_start_time;
        #pragma omp critical
        {
            std::cout << "[Поток " << tid << " завершил работу за " << thread_time << " секунд]\n";
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
}
