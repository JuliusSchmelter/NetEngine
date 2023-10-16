cmake_minimum_required(VERSION 3.0.0)
project(BigEarthNet)

# build netlib
add_subdirectory(NetEngine)

# make exe
add_executable(NetEngineTest main.cpp)

# include and link netlib
target_include_directories(NetEngineTest PRIVATE NetEngine)
target_link_libraries(NetEngineTest PRIVATE NetEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wall -Wextra -pedantic -Werror -O3)
