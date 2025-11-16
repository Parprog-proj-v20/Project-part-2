cmake_minimum_required(VERSION 3.13)
project(Project-part-2)

set(CMAKE_CXX_STANDARD 17)

# Явно указываем абсолютные пути
set(MAIN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Matrix.cpp
)

set(TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/MatrixTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Matrix.cpp
)

find_package(OpenMP REQUIRED)

# Основное приложение
add_executable(Project-part-2 ${MAIN_SOURCES})
target_link_libraries(Project-part-2 OpenMP::OpenMP_CXX)

# Явно указываем пути include
target_include_directories(Project-part-2
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(Project-part-2 PRIVATE "/openmp")
endif()

# Тесты
add_executable(matrix_tests ${TEST_SOURCES})
target_link_libraries(matrix_tests OpenMP::OpenMP_CXX)

target_include_directories(matrix_tests
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
)

if(MSVC)
    target_compile_options(matrix_tests PRIVATE "/openmp")
endif()