void MatrixTest::testMultiplicationCorrectness() {
    std::cout << "Проверка корректности умножения матриц 4x4" << std::endl;
    
    Matrix matrix(4);
    
    matrix.A = {{1, 2, 3, 4},
                {5, 6, 7, 8},
                {9, 10, 11, 12},
                {13, 14, 15, 16}};
    
    matrix.B = {{1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0},
                {0, 0, 0, 1}};

    std::vector<std::vector<int>> expected = {{1, 2, 3, 4},
                                             {5, 6, 7, 8},
                                             {9, 10, 11, 12},
                                             {13, 14, 15, 16}};

    matrix.multiplyLinear();
    assert(areMatricesEqual(matrix.C, expected));

    std::vector<std::string> schedules = {"static", "dynamic", "guided"};
    for (const auto& schedule : schedules) {
        matrix.multiplyParallel(2, schedule);
        assert(areMatricesEqual(matrix.C, expected));
    }
}


void MatrixTest::testDifferentSizes() {
    std::vector<int> testSizes = {2, 10, 50};
    
    for (int size : testSizes) {
        std::cout << "Тестирование размера " << size << "x" << size << "..." << std::endl;
        
        Matrix matrix(size);
        matrix.initialize();
        
        auto expected = simpleMultiply(matrix.A, matrix.B);
        
        matrix.multiplyLinear();
        assert(areMatricesEqual(matrix.C, expected));
        
        matrix.multiplyParallel(4, "static");
        assert(areMatricesEqual(matrix.C, expected));
    }
}


void MatrixTest::testEdgeCases() {
    std::cout << "Тестирование граничных случаев" << std::endl;
    
    // Тест 1: Матрица 1x1
    Matrix matrix1(1);
    matrix1.A = {{5}};
    matrix1.B = {{3}};
    matrix1.multiplyParallel(2, "static");
    assert(matrix1.C[0][0] == 15);

    // Тест 2: Нулевая матрица
    Matrix matrix2(3);
    matrix2.A = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    matrix2.B = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    std::vector<std::vector<int>> expected = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    matrix2.multiplyParallel(2, "static");
    assert(areMatricesEqual(matrix2.C, expected));

    // Тест 3: Единичная матрица
    Matrix matrix3(3);
    matrix3.A = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    matrix3.B = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    matrix3.multiplyParallel(2, "static");
    assert(areMatricesEqual(matrix3.C, matrix3.A));
}


void MatrixTest::testPerformance() {
    std::cout << "Тестирование производительности" << std::endl;
    
    Matrix matrix(300);
    matrix.initialize();
    
    double singleThreadTime = matrix.multiplyParallel(1, "static");
    
    std::vector<int> threadCounts = {2, 4};
    for (int threads : threadCounts) {
        double multiThreadTime = matrix.multiplyParallel(threads, "static");
        validateSpeedup(singleThreadTime, multiThreadTime, threads, "static scheduling");
        assert(multiThreadTime <= singleThreadTime * 1.5);
    }
}


void MatrixTest::testSchedulingTypes() {
    std::cout << "Сравнение типов планирования" << std::endl;
    
    Matrix matrix(200);
    matrix.initialize();
    
    std::vector<std::string> schedules = {"static", "dynamic", "guided"};
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



void MatrixTest::testLinearVsParallel() {
    std::cout << "Сравнение линейного и параллельного умножения" << std::endl;
    
    Matrix matrix(100);
    matrix.initialize();
    
    matrix.multiplyLinear();
    auto linearResult = matrix.C;
    
    matrix.multiplyParallel(2, "static");
    auto parallelResult = matrix.C;
    
    assert(areMatricesEqual(linearResult, parallelResult));
}


void MatrixTest::testKnownMatrices() {
    std::cout << "Тестирование на известных матрицах" << std::endl;
    
    Matrix matrix(2);
    matrix.A = {{1, 2}, {3, 4}};
    matrix.B = {{2, 0}, {1, 2}};
    
    std::vector<std::vector<int>> expected = {{4, 4}, {10, 8}};
    
    matrix.multiplyParallel(2, "static");
    assert(areMatricesEqual(matrix.C, expected));
}


bool MatrixTest::areMatricesEqual(const std::vector<std::vector<int>>& matrix1, 
                                 const std::vector<std::vector<int>>& matrix2) {
    if (matrix1.size() != matrix2.size()) return false;
    
    for (size_t i = 0; i < matrix1.size(); i++) {
        if (matrix1[i].size() != matrix2[i].size()) return false;
        for (size_t j = 0; j < matrix1[i].size(); j++) {
            if (matrix1[i][j] != matrix2[i][j]) {
                return false;
            }
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


