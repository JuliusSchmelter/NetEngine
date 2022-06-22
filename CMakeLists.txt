cmake_minimum_required(VERSION 3.0.0)
project(BigEarthNet)

# build netlib
add_subdirectory(netlib)

# make exe
add_executable(BigEarthNet main.cpp)

# include and link netlib
target_include_directories(BigEarthNet PRIVATE netlib)
target_link_libraries(BigEarthNet PRIVATE netlib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wall -Wextra -pedantic -Werror -O3)
