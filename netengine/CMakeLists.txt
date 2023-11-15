cmake_minimum_required(VERSION 3.18)
project(netengine LANGUAGES CXX CUDA)

# Set C++ version.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable warnings, optimizations and position-independent code for dynamic linking.
add_compile_options(-Wall -Wextra -pedantic -Werror -O3 -fPIC)

# Build Python module using pybind11.
add_subdirectory(pybind11)
pybind11_add_module(netengine netengine.cpp)

# Build NetEngine.
add_subdirectory(NetEngine)

# Link NetEngine to netengine python module.
target_link_libraries(netengine PRIVATE NetEngine)

# Set CUDA architecture
set_target_properties(netengine PROPERTIES CUDA_ARCHITECTURES 75)