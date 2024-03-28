#include <cassert>
#include <curand_kernel.h>
#include <iomanip>
#include <iostream>

#include "Exceptions.h"
#include "Net.h"

// CUDA kernel to set random float values in device memory.
// Target matrix is m x n.
__global__ void set_random(float* data, uint32_t m, uint32_t n, size_t seed, float range) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize random number generator.
    curandState curand_state;
    curand_init(seed, row, 0, &curand_state);

    if (row < m)
        for (uint32_t col = 0; col < n; col++)
            data[row * n + col] = (curand_uniform(&curand_state) * 2.0f - 1.0f) * range;
}

//--------------------------------------------------------------------------------------------------
// Constructor
//--------------------------------------------------------------------------------------------------
NetEngine::Net::Net(const std::vector<uint32_t>& layout, float eta, float eta_bias, bool try_cuda)
    : m_layout(layout), m_eta(eta), m_eta_bias(eta_bias) {
    // Check minimum number of layers.
    if (layout.size() < 3)
        throw NetEngine::NotEnoughLayers(layout.size());

    // Seed random number generator to get new random numbers each time.
    std::srand((unsigned)std::time(0));

    // Check if CUDA capable device is available.
    int cuda_device;
    if (try_cuda && cudaGetDevice(&cuda_device) == cudaSuccess) {
        m_cuda_enabled = true;

        // Initialize weight matrices, add one column for bias.
        for (size_t i = 0; i < layout.size() - 1; i++) {
            float* weights;
            uint32_t rows = layout[i + 1];
            uint32_t cols = layout[i] + 1;

            TRY_CUDA(cudaMalloc(&weights, rows * cols * sizeof(float)));

            // Set weight matrix to random values in range [+- 1/sqrt(number of neurons in layer)].
            size_t threads_per_block = BLOCK_SIZE;
            size_t n_blocks = (rows + threads_per_block - 1) / threads_per_block;
            set_random<<<n_blocks, threads_per_block>>>(weights, rows, cols, i,
                                                        1.0f / sqrt(cols - 1));

            TRY_CUDA(cudaDeviceSynchronize());

            CudaMatrix cuda_matrix = {weights, rows, cols};
            m_weights_cuda.push_back(cuda_matrix);
        }
    } else {
        m_cuda_enabled = false;

        // Initialize weight matrices, add column for bias
        for (size_t i = 0; i < layout.size() - 1; i++)
            m_weights_eigen.push_back(Eigen::MatrixXf(layout[i + 1], layout[i] + 1));

        for (auto& i : m_weights_eigen) {
            // Set weight matrix to random values in range [-1, 1].
            i.setRandom();

            // Change range to [+- 1/sqrt(number of neurons in layer)].
            i / sqrt(i.cols() - 1);
        }
    }
}

//--------------------------------------------------------------------------------------------------
// Destructor
//--------------------------------------------------------------------------------------------------
NetEngine::Net::~Net() {
    if (m_cuda_enabled)
        for (auto& i : m_weights_cuda)
            TRY_CUDA(cudaFree(i.data));
}

//--------------------------------------------------------------------------------------------------
// Getters and setters.
//--------------------------------------------------------------------------------------------------
float NetEngine::Net::get_eta() {
    return m_eta;
}
void NetEngine::Net::set_eta(float eta) {
    m_eta = eta;
}
float NetEngine::Net::get_eta_bias() {
    return m_eta_bias;
}
void NetEngine::Net::set_eta_bias(float eta_bias) {
    m_eta_bias = eta_bias;
}
bool NetEngine::Net::cuda_enabled() {
    return m_cuda_enabled;
}

//--------------------------------------------------------------------------------------------------
// Return info string.
//--------------------------------------------------------------------------------------------------
std::string NetEngine::Net::info_string() {
    std::stringstream output;

    output << "layout: ";
    for (size_t i = 0; i < m_layout.size(); i++) {
        output << m_layout[i];
        if (i < m_layout.size() - 1)
            output << " | ";
    }

    output << "\nparameters: " << n_parameters() << "\neta: " << m_eta
           << "\neta_bias: " << m_eta_bias;

    return output.str();
}

//--------------------------------------------------------------------------------------------------
// Get number of parameters.
//--------------------------------------------------------------------------------------------------
size_t NetEngine::Net::n_parameters() {
    size_t n = 0;

    if (m_cuda_enabled)
        for (auto& i : m_weights_cuda)
            n += i.rows * i.cols;
    else
        for (auto& i : m_weights_eigen)
            n += i.rows() * i.cols();

    return n;
}

//--------------------------------------------------------------------------------------------------
// Run one sample through the network.
//--------------------------------------------------------------------------------------------------
std::vector<float> NetEngine::Net::run(const std::vector<float>& sample) {
    // Delegate to Eigen or CUDA.
    if (m_cuda_enabled)
        return run_cuda(sample);
    else
        return run_eigen(sample);
}

//--------------------------------------------------------------------------------------------------
// Train the network using backpropagation.
//--------------------------------------------------------------------------------------------------
size_t NetEngine::Net::train(const std::vector<std::vector<float>>& samples,
                             const std::vector<std::vector<uint8_t>>& labels, size_t n_samples,
                             size_t start_pos) {
    // Delegate to Eigen or CUDA.
    if (m_cuda_enabled)
        return train_cuda(samples, labels, n_samples, start_pos);
    else
        return train_eigen(samples, labels, n_samples, start_pos);
}

//--------------------------------------------------------------------------------------------------
// Test accuracy.
//--------------------------------------------------------------------------------------------------
float NetEngine::Net::test(const std::vector<std::vector<float>>& samples,
                           const std::vector<std::vector<uint8_t>>& labels) {
    // Delegate to Eigen or CUDA.
    if (m_cuda_enabled)
        return test_cuda(samples, labels);
    else
        return test_eigen(samples, labels);
}
