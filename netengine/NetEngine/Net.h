#ifndef NETENGINE_NET_H
#define NETENGINE_NET_H

// Recommended for using Eigen on CUDA.
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#include "extern/Eigen/Dense"
#include <thread>
#include <vector>

namespace NetEngine {
    class Net {
    private:
        const std::vector<size_t> m_layout;
        std::vector<Eigen::MatrixXf> m_weights;
        float m_eta;
        float m_eta_bias;

        void get_weight_mods(const std::vector<Eigen::Map<const Eigen::VectorXf>>& samples,
                             const std::vector<Eigen::Map<const Eigen::VectorX<uint32_t>>>& labels,
                             std::vector<Eigen::MatrixXf>& weight_mods);

        void test_thread(const std::vector<std::vector<float>>& samples,
                         const std::vector<std::vector<uint32_t>>& labels, float& output,
                         size_t start_pos, size_t batch_size, float subset, float threshold);

        void test_thread_cpu(const std::vector<std::vector<float>>& samples,
                             const std::vector<std::vector<uint32_t>>& labels, float& output,
                             size_t start_pos, size_t batch_size, float subset, float threshold);

    public:
        Net(const std::vector<size_t>& layout, float eta, float eta_bias);
        ~Net() = default;

        float get_eta();
        void set_eta(float eta);
        float get_eta_bias();
        void set_eta_bias(float eta_bias);

        std::string info_string();
        size_t n_parameters();
        void set_random();

        // Run one sample on the net.
        std::vector<float> run(const std::vector<float>& sample);

        // Train net on GPU.
        void train(const std::vector<std::vector<float>>& samples,
                   const std::vector<std::vector<uint32_t>>& labels, size_t n_batches,
                   size_t batch_size, size_t start_pos = 0,
                   size_t n_threads = std::thread::hardware_concurrency());

        // Train net on CPU.
        void train_on_cpu(const std::vector<std::vector<float>>& samples,
                          const std::vector<std::vector<uint32_t>>& labels, size_t n_batches,
                          size_t batch_size, size_t start_pos = 0,
                          size_t n_threads = std::thread::hardware_concurrency());

        // Test accuracy on GPU.
        float test(const std::vector<std::vector<float>>& samples,
                   const std::vector<std::vector<uint32_t>>& labels, float subset = 100.0f,
                   float threshold = NAN, size_t n_threads = std::thread::hardware_concurrency());

        // Test accuracy on CPU.
        float test_on_cpu(const std::vector<std::vector<float>>& samples,
                          const std::vector<std::vector<uint32_t>>& labels, float subset = 100.0f,
                          float threshold = NAN,
                          size_t n_threads = std::thread::hardware_concurrency());
    };
}

#endif
