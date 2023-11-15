set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_library(NetEngine 
src/Net.cpp
src/train.cu
src/test.cu
)