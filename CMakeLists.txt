cmake_minimum_required(VERSION 3.6)
project(WissRech2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++11 -msse4 ")

set(SOURCE_FILES main.cpp)
add_executable(WissRech1 ${SOURCE_FILES})