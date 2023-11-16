# rm -rf build && cmake . -Bbuild && (cd build && make) && build/NetEngineTest

cmake_minimum_required(VERSION 3.25)
project(NetEngineTest LANGUAGES CXX CUDA)

# Set C++ version.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Suppress unnecessary nvcc warnings.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -diag-suppress 20012" )

# Build lib.
add_subdirectory(netengine/NetEngine)

# Make exe.
add_executable(NetEngineTest main.cpp)

# Include and link lib.
target_include_directories(NetEngineTest PRIVATE NetEngine)
target_link_libraries(NetEngineTest PRIVATE NetEngine)

# Enable warnings and optimizations.
add_compile_options(-Wall -Wextra -pedantic -Werror -O3)

# Set CUDA architecture.
set_target_properties(NetEngineTest PROPERTIES CUDA_ARCHITECTURES 75)
