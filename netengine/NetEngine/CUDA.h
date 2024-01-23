#ifndef NETENGINE_CUDA_H
#define NETENGINE_CUDA_H

// CUDA block size.
#define BLOCK_SIZE 8

// Error handling for CUDA functions.
#define TRY_CUDA(cuda_function)                                                                    \
    {                                                                                              \
        cudaError_t res = cuda_function;                                                           \
        if (res != cudaSuccess) {                                                                  \
            std::cerr << "CUDA Error: " << cudaGetErrorString(res) << " @ " << __FILE__ << " ("    \
                      << __LINE__ << ")" << std::endl;                                             \
            abort();                                                                               \
        }                                                                                          \
    }

// CUDA matrix in device memory.
namespace NetEngine {
    struct CudaMatrix {
        float* data;
        uint32_t rows;
        uint32_t cols;
    };
}

#endif // NETENGINE_CUDA_H