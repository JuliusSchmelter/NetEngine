cmake_minimum_required(VERSION 3.0.0)
project(BigEarthNet VERSION 1.0)

include(CTest)
enable_testing()

add_executable(BigEarthNet main.cpp)

add_subdirectory(libs/eigen-3.4.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
