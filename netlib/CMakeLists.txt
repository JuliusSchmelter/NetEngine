cmake_minimum_required(VERSION 3.0)
project(BigEarthNet)

add_library(netlib 
    src/net.cpp
    )

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_subdirectory(extern/eigen-3.4.0)
target_include_directories(netlib PRIVATE extern/eigen-3.4.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
