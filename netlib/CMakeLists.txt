cmake_minimum_required(VERSION 3.0)
include_directories(extern)

add_library(netlib 
src/net.cpp
src/train.cpp
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)