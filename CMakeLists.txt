cmake_minimum_required(VERSION 3.10)
project(BigEarthNet)

add_subdirectory(netlib)

add_executable(BigEarthNet main.cpp)

target_link_libraries(BigEarthNet netlib)
target_include_directories(BigEarthNet PRIVATE netlib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wall -Wextra -pedantic -Werror)