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
