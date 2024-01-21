#ifndef NETENGINE_NET_H
#define NETENGINE_NET_H

#include <cassert>
#include <iostream>
#include <thread>
#include <vector>

// CUDA block size.
#define BLOCK_SIZE 16

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

namespace NetEngine {
    struct CudaMatrix {
        float* data;
        uint32_t rows;
        uint32_t cols;
    };

    class Net {
    private:
        const std::vector<uint32_t> m_layout;
        std::vector<CudaMatrix> m_weights;
        float m_eta;
        float m_eta_bias;

        // Run one sample on the net. Internal function, expects pointers to device memory.
        void run_cuda(float* sample, float** results);

    public:
        Net(const std::vector<uint32_t>& layout, float eta, float eta_bias);
        ~Net();

        float get_eta();
        void set_eta(float eta);
        float get_eta_bias();
        void set_eta_bias(float eta_bias);

        std::string info_string();
        size_t n_parameters();

        // Run one sample on the net.
        std::vector<float> run(const std::vector<float>& sample);

        // Train net. Returns last position in samples.
        size_t train(const std::vector<std::vector<float>>& samples,
                     const std::vector<std::vector<uint32_t>>& labels, size_t n_samples,
                     size_t start_pos = 0);

        // Test accuracy.
        float test(const std::vector<std::vector<float>>& samples,
                   const std::vector<std::vector<uint32_t>>& labels);
    };
}

#endif
