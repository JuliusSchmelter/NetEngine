#include "Net.h"

// CUDA kernel for matrix multiplication.
// A is m x n, B is n x p, C is m x p.
__global__ void matmul(float* A, float* B, float* C, uint32_t m, uint32_t n, uint32_t p) {
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        float sum = 0;
        for (uint32_t i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * p + col];
        }
        C[row * p + col] = sum;
    }
}

//--------------------------------------------------------------------------------------------------
// Run one sample through the network.
//--------------------------------------------------------------------------------------------------
std::vector<float> NetEngine::Net::run(const std::vector<float>& sample) {
    // check dimensions
    if (sample.size() != m_layout.front())
        throw NetEngine::DimensionError(sample.size(), m_layout.front());

    // Copy sample to unified memory.
    float* sample_cuda;
    TRY_CUDA(cudaMallocManaged(&sample_cuda, sample.size() * sizeof(float)));
    memcpy(sample_cuda, sample.data(), sample.size() * sizeof(float));

    // Allocate memory for results.
    float* results_cuda[m_weights.size()];
    for (size_t i = 0; i < m_weights.size(); i++) {
        TRY_CUDA(cudaMallocManaged(&results_cuda[i], m_weights[i].cols * sizeof(float)));
    }

    // Run sample on network.
    run_cuda(sample_cuda, results_cuda);

    // Copy output to vector.
    std::vector<float> output(results_cuda[m_weights.size()],
                              results_cuda[m_weights.size()] + m_layout.back());

    // Free memory.
    TRY_CUDA(cudaFree(sample_cuda));
    for (size_t i = 0; i < m_weights.size(); i++) {
        TRY_CUDA(cudaFree(results_cuda[i]));
    }

    return output;
}

//--------------------------------------------------------------------------------------------------
// Run one sample on the net. Internal function, expects pointers to CUDA unified memory.
//--------------------------------------------------------------------------------------------------
// CUDA kernel to feed forward one layer.
// input is m x 1, weights is n x (m + 1) (for bias), output is n x 1.
__global__ void feed_forward(float* input, float* weights, float* output, uint32_t m, uint32_t n) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        // Note: last column is bias.
        float sum = weights[row * (m + 1) + m];
        for (uint32_t col = 0; col < m - 1; col++) {
            sum += input[col] * weights[row * (m + 1) + col];
        }
        // Apply sigmoid function.
        output[row] = 0.5f + 0.5f * sum / (1.0f + fabsf(sum));
    }
}

void NetEngine::Net::run_cuda(const float* sample, float** results) {
    // Run first layer.
    size_t threadsPerBlock = BLOCK_SIZE * BLOCK_SIZE;
    size_t numBlocks = (m_weights[0].cols + threadsPerBlock - 1) / threadsPerBlock;
    feed_forward<<<numBlocks, threadsPerBlock>>>(sample, m_weights[0].data, results[0],
                                                 m_weights[0].cols - 1, m_weights[0].rows);
}

std::vector<float> NetEngine::Net::run(const std::vector<float>& sample) {
    // check dimensions
    if (sample.size() != m_layout.front())
        throw NetEngine::DimensionError(sample.size(), m_layout.front());

    // get Eigen vector for input, this does not copy the data
    Eigen::Map<const Eigen::VectorXf> input(sample.data(), sample.size());

    // run first layer
    // note: last column is bias
    Eigen::VectorXf a = (m_weights[0].leftCols(m_weights[0].cols() - 1) * input +
                         m_weights[0].col(m_weights[0].cols() - 1))
                            .unaryExpr([](float x) { return 0.5f + 0.5f * x / (1.0f + fabsf(x)); });

    // run remaining layers
    for (size_t i = 1; i < m_weights.size(); i++) {
        // note: eval() forces evaluation before new values are assigned to a
        a = ((m_weights[i].leftCols(m_weights[i].cols() - 1) * a).eval() +
             m_weights[i].col(m_weights[i].cols() - 1))
                .unaryExpr([](float x) { return 0.5f + 0.5f * x / (1.0f + fabsf(x)); });
    }

    return std::vector<float>(a.data(), a.data() + a.size());
}