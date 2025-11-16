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
