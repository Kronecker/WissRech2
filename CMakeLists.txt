cmake_minimum_required(VERSION 3.8)
project(WissRech2)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES Task4/Debricated/main.cu Task4/testMe.cpp Task3/testme.cpp Task3/testMe2.cpp Task4/mmain.cu Task4/aufg13a.cpp Task4/aufg13b.cpp)
add_executable(WissRech2 ${SOURCE_FILES})