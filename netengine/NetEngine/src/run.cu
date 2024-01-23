#include "Exceptions.h"
#include "Net.h"

// CUDA kernel to feed forward one layer.
// input is m x 1, weights is n x (m + 1) (for bias), output is n x 1.
__global__ void feed_forward(float* input, float* weights, float* output, uint32_t m, uint32_t n) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        // Note: last column is bias.
        float sum = weights[row * (m + 1) + m];
        for (uint32_t col = 0; col < m - 1; col++)
            sum += input[col] * weights[row * (m + 1) + col];

        // Apply sigmoid function.
        output[row] = 0.5f + 0.5f * sum / (1.0f + fabsf(sum));
    }
}

//--------------------------------------------------------------------------------------------------
// Run one sample through the network using CUDA.
//--------------------------------------------------------------------------------------------------
std::vector<float> NetEngine::Net::run_cuda(const std::vector<float>& sample) {
    // Check dimensions.
    if (sample.size() != m_layout.front())
        throw NetEngine::DimensionError(sample.size(), m_layout.front());

    // Copy sample to device memory.
    float* sample_cuda;
    TRY_CUDA(cudaMalloc(&sample_cuda, sample.size() * sizeof(float)));
    TRY_CUDA(cudaMemcpy(sample_cuda, sample.data(), sample.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

    // Allocate memory for results.
    float* results_cuda[m_weights_cuda.size()];
    for (size_t i = 0; i < m_weights_cuda.size(); i++) {
        TRY_CUDA(cudaMalloc(&results_cuda[i], m_weights_cuda[i].cols * sizeof(float)));
    }

    // Run sample on network.
    run_cuda_dev_ptrs(sample_cuda, results_cuda);

    // Copy output to vector.
    std::vector<float> output(m_layout.back());
    TRY_CUDA(cudaMemcpy(output.data(), results_cuda[m_weights_cuda.size() - 1],
                        m_layout.back() * sizeof(float), cudaMemcpyDeviceToHost));

    // Free memory.
    TRY_CUDA(cudaFree(sample_cuda));
    for (size_t i = 0; i < m_weights_cuda.size(); i++) {
        TRY_CUDA(cudaFree(results_cuda[i]));
    }

    return output;
}

//--------------------------------------------------------------------------------------------------
// Run one sample on the net. Internal function, expects pointers to device memory.
//--------------------------------------------------------------------------------------------------
void NetEngine::Net::run_cuda_dev_ptrs(float* sample, float** results) {
    // Run first layer.
    size_t threads_per_block = BLOCK_SIZE * BLOCK_SIZE;
    size_t n_blocks = (m_weights_cuda[0].rows + threads_per_block - 1) / threads_per_block;

    feed_forward<<<n_blocks, threads_per_block>>>(sample, m_weights_cuda[0].data, results[0],
                                                  m_weights_cuda[0].cols - 1,
                                                  m_weights_cuda[0].rows);

    TRY_CUDA(cudaDeviceSynchronize());

    // Run remaining layers.
    for (size_t i = 1; i < m_weights_cuda.size(); i++) {
        n_blocks = (m_weights_cuda[i].rows + threads_per_block - 1) / threads_per_block;

        feed_forward<<<n_blocks, threads_per_block>>>(results[i - 1], m_weights_cuda[i].data,
                                                      results[i], m_weights_cuda[i].cols - 1,
                                                      m_weights_cuda[i].rows);

        TRY_CUDA(cudaDeviceSynchronize());
    }
}
