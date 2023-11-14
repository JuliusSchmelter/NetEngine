cmake_minimum_required(VERSION 3.25)
project(NetEngineTest LANGUAGES CXX CUDA)

# CUDA options
# set(CMAKE_CUDA_ARCHITECTURES 75)
# set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# find_package(CUDAToolkit)
# set(CMAKE_CUDA_STANDARD 14)

# set_target_properties(NetEngineTest PROPERTIES CUDA_ARCHITECTURES 75)


# Build lib.
add_subdirectory(netengine/NetEngine)

# Make exe.
add_executable(NetEngineTest main.cpp)

# Include and link lib.
target_include_directories(NetEngineTest PRIVATE NetEngine)
target_link_libraries(NetEngineTest PRIVATE NetEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wall -Wextra -pedantic -Werror -O3)
