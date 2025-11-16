cmake_minimum_required(VERSION 3.13)
project(Project-part-2)

set(CMAKE_CXX_STANDARD 17)

set(SOURCES
    src/main.cpp
    src/Matrix.cpp
)

find_package(OpenMP REQUIRED)

add_executable(Project-part-2 ${SOURCES})

target_link_libraries(Project-part-2 OpenMP::OpenMP_CXX)

target_include_directories(Project-part-2
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
)