set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_library(NetEngine 
src/Net.cpp
src/train.cpp
src/train.cu
src/test.cpp
src/test.cu
)