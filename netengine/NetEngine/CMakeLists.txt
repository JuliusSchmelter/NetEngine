set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_library(NetEngine 
src/Net.cu
src/run.cu
src/test.cu
src/train.cu
src/run.cpp
src/test.cpp
src/train.cpp
)