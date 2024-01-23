#ifndef NETENGINE_NET_H
#define NETENGINE_NET_H

#include "extern/Eigen/Dense"
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>

#include "CUDA.h"

namespace NetEngine {
    class Net {
    public:
        Net(const std::vector<uint32_t>& layout, float eta, float eta_bias, bool try_cuda = true);
        ~Net();

        float get_eta();
        void set_eta(float eta);
        float get_eta_bias();
        void set_eta_bias(float eta_bias);
        bool cuda_enabled();

        std::string info_string();
        size_t n_parameters();

        // Run one sample on the net.
        std::vector<float> run(const std::vector<float>& sample);

        // Train net. Returns last position in samples.
        size_t train(const std::vector<std::vector<float>>& samples,
                     const std::vector<std::vector<uint8_t>>& labels, size_t n_samples,
                     size_t start_pos = 0);

        // Test accuracy.
        float test(const std::vector<std::vector<float>>& samples,
                   const std::vector<std::vector<uint8_t>>& labels);

    private:
        const std::vector<uint32_t> m_layout;
        float m_eta;
        float m_eta_bias;

        // If a CUDA device is available, weights are stored in device memory and CUDA is used for
        // computations. Otherwise, Eigen matrices on the CPU are used as a fallback.
        bool m_cuda_enabled;
        std::vector<CudaMatrix> m_weights_cuda;
        std::vector<Eigen::MatrixXf> m_weights_eigen;

        // Internal functions for Eigen or CUDA.
        std::vector<float> run_cuda(const std::vector<float>& sample);
        std::vector<float> run_eigen(const std::vector<float>& sample);

        size_t train_cuda(const std::vector<std::vector<float>>& samples,
                          const std::vector<std::vector<uint8_t>>& labels, size_t n_samples,
                          size_t start_pos);
        size_t train_eigen(const std::vector<std::vector<float>>& samples,
                           const std::vector<std::vector<uint8_t>>& labels, size_t n_samples,
                           size_t start_pos);
        void train_worker_eigen(const std::vector<std::vector<float>>& samples,
                                const std::vector<std::vector<uint8_t>>& labels, size_t n_samples,
                                size_t start_pos, std::vector<Eigen::MatrixXf>& weight_mods,
                                std::vector<Eigen::VectorXf>& a, std::vector<Eigen::VectorXf>& z,
                                std::vector<Eigen::VectorXf>& deltas);

        float test_cuda(const std::vector<std::vector<float>>& samples,
                        const std::vector<std::vector<uint8_t>>& labels);
        float test_eigen(const std::vector<std::vector<float>>& samples,
                         const std::vector<std::vector<uint8_t>>& labels);
        void test_worker_eigen(float& output, const std::vector<std::vector<float>>& samples,
                               const std::vector<std::vector<uint8_t>>& labels, size_t start_pos,
                               size_t batch_size);

        // Run one sample on the net. Internal function, expects pointers to device memory.
        void run_cuda_dev_ptrs(float* sample, float** results);
    };
}

#endif // NETENGINE_NET_H