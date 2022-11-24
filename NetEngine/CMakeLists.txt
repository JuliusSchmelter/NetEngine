cmake_minimum_required(VERSION 3.0)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_library(NetEngine 
src/Net.cpp
src/train.cpp
src/test.cpp
)