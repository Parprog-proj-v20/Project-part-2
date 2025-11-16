# Тесты
add_executable(matrix_tests test_main.cpp MatrixTest.cpp ../Matrix.cpp)
target_link_libraries(matrix_tests OpenMP::OpenMP_CXX)

# Путь к родительскому каталогу для включения заголовков
target_include_directories(matrix_tests PRIVATE ..)
