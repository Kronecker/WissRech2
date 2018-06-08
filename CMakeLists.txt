cmake_minimum_required(VERSION 3.8)
project(WissRech2)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES Task4/noCuda.cpp)
add_executable(WissRech2 ${SOURCE_FILES})