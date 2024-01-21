#include "DevTools.h"
#include "Exceptions.h"
#include "Net.h"

// CUDA kernel to feed forward one layer and also store the weighted input.
// input is m x 1, weights is n x (m + 1) (for bias), a and z are n x 1.
__global__ void feed_forward_training(float* input, float* weights, float* a, float* z, uint32_t m,
                                      uint32_t n) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        // Note: last column is bias.
        float sum = weights[row * (m + 1) + m];
        for (uint32_t col = 0; col < m - 1; col++)
            sum += input[col] * weights[row * (m + 1) + col];

        // Store the weighted input.
        z[row] = sum;

        // Apply sigmoid function.
        a[row] = 0.5f + 0.5f * sum / (1.0f + fabsf(sum));
    }
}

// CUDA kernel to calculate the deltas for the output layer.
// n is the size of the output layer.
__global__ void calc_deltas_output(uint32_t* label, float* a, float* z, float* deltas, uint32_t n) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        // Get loss by comparing activation with label, then take hadamard product with
        // first derivative of sigmoid of weighted input.
        deltas[row] = (a[row] - (float)label[row]) / (1.0f + powf(fabsf(z[row]), 2));
    }
}

// CUDA kernel to calculate the deltas for first and hidden layers.
// n x m is the size of weights_next.
__global__ void calc_deltas(float* deltas, float* deltas_next, float* weights_next, float* z,
                            uint32_t n, uint32_t m) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n) {
        // Backpropagate delta of following layer.
        float sum = 0;
        for (uint32_t i = 0; i < n; i++)
            // Note: row is the row of deltas, and therefore the column of weights_next.
            // We are calculating transpose(weights_next) * deltas_next.
            sum += deltas_next[i] * weights_next[(i * m) + row];

        // Take hadamard product with first derivative of sigmoid of weighted input.
        deltas[row] = sum / (1.0f + powf(fabsf(z[row]), 2));
    }
}

// CUDA kernel to update the weights.
// n x m is the size of weights.
__global__ void update_weights(float* weights, float* deltas, float* a, float eta, float eta_bias,
                               uint32_t n, uint32_t m) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < m - 1) {
        // Update weight.
        weights[(row * m) + col] -= eta * deltas[row] * a[col];
    } else if (row < n && col == m - 1) {
        // Update bias.
        weights[(row * m) + col] -= eta_bias * deltas[row];
    }
}

//--------------------------------------------------------------------------------------------------
// Train the network using backpropagation.
//--------------------------------------------------------------------------------------------------
size_t NetEngine::Net::train(const std::vector<std::vector<float>>& samples,
                             const std::vector<std::vector<uint32_t>>& labels, size_t n_samples,
                             size_t start_pos) {
    // Check for bad inputs.
    if (samples[0].size() != m_layout.front())
        throw NetEngine::DimensionError(samples[0].size(), m_layout.front());

    if (labels[0].size() != m_layout.back())
        throw NetEngine::DimensionError(labels[0].size(), m_layout.back());

    if (samples.size() != labels.size())
        throw NetEngine::SetSizeError(samples.size(), labels.size());

    if (start_pos >= samples.size())
        start_pos = 0;

    // Allocate device memory for sample and label.
    float* d_sample;
    uint32_t* d_label;
    cudaMalloc(&d_sample, sizeof(float) * m_layout.front());
    cudaMalloc(&d_label, sizeof(uint32_t) * m_layout.back());

    // Allocate device memory for activation values and deltas.
    float* z[m_weights.size()];
    float* a[m_weights.size()];
    float* deltas[m_weights.size()];
    for (size_t i = 0; i < m_weights.size(); i++) {
        cudaMalloc(&z[i], sizeof(float) * m_layout[i + 1]);
        cudaMalloc(&a[i], sizeof(float) * m_layout[i + 1]);
        cudaMalloc(&deltas[i], sizeof(float) * m_layout[i + 1]);
    }

    // Position in samples and labels.
    size_t pos = start_pos;

    // Iterate samples.
    for (size_t s = 0; s < n_samples; s++) {
        // Copy sample and label to device.
        cudaMemcpy(d_sample, samples[pos].data(), sizeof(float) * m_layout.front(),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_label, labels[pos].data(), sizeof(uint32_t) * m_layout.back(),
                   cudaMemcpyHostToDevice);

        // Feed forward first layer.
        size_t threads_per_block = BLOCK_SIZE * BLOCK_SIZE;
        size_t n_blocks = (m_weights[0].rows + threads_per_block - 1) / threads_per_block;
        feed_forward_training<<<n_blocks, threads_per_block>>>(
            d_sample, m_weights[0].data, a[0], z[0], m_weights[0].cols - 1, m_weights[0].rows);

        // Feed forward remaining layers.
        for (size_t i = 1; i < m_weights.size(); i++) {
            n_blocks = (m_weights[i].rows + threads_per_block - 1) / threads_per_block;
            feed_forward_training<<<n_blocks, threads_per_block>>>(
                a[i - 1], m_weights[i].data, a[i], z[i], m_weights[i].cols - 1, m_weights[i].rows);
        }

        // Calculate deltas for output layer.
        n_blocks = (m_layout.back() + threads_per_block - 1) / threads_per_block;
        calc_deltas_output<<<n_blocks, threads_per_block>>>(
            d_label, a[m_weights.size() - 1], z[m_weights.size() - 1], deltas[m_weights.size() - 1],
            m_layout.back());

        TRY_CUDA(cudaDeviceSynchronize());

        // Calculate deltas for remaining layers.
        for (size_t i = m_weights.size() - 1; i > 0; i--) {
            n_blocks = (m_layout[i - 1] + threads_per_block - 1) / threads_per_block;
            calc_deltas<<<n_blocks, threads_per_block>>>(deltas[i - 1], deltas[i],
                                                         m_weights[i].data, z[i - 1],
                                                         m_weights[i].rows, m_weights[i].cols);

            TRY_CUDA(cudaDeviceSynchronize());
        }

        // Update weights of input layer.
        dim3 threads_per_block_2d(BLOCK_SIZE, BLOCK_SIZE);
        dim3 n_blocks_2d((m_weights[0].rows + threads_per_block_2d.x - 1) / threads_per_block_2d.x,
                         (m_weights[0].cols + threads_per_block_2d.y - 1) / threads_per_block_2d.y);

        update_weights<<<n_blocks_2d, threads_per_block_2d>>>(m_weights[0].data, deltas[0],
                                                              d_sample, m_eta, m_eta_bias,
                                                              m_weights[0].rows, m_weights[0].cols);

        TRY_CUDA(cudaDeviceSynchronize());

        // Update weights of remaining layers.
        for (size_t i = 1; i < m_weights.size(); i++) {
            n_blocks_2d =
                dim3((m_weights[i].rows + threads_per_block_2d.x - 1) / threads_per_block_2d.x,
                     (m_weights[i].cols + threads_per_block_2d.y - 1) / threads_per_block_2d.y);

            update_weights<<<n_blocks_2d, threads_per_block_2d>>>(
                m_weights[i].data, deltas[i], a[i - 1], m_eta, m_eta_bias, m_weights[i].rows,
                m_weights[i].cols);

            TRY_CUDA(cudaDeviceSynchronize());
        }

        pos = (pos + 1) % samples.size();
    }

    // Free device memory.
    cudaFree(d_sample);
    cudaFree(d_label);
    for (size_t i = 0; i < m_weights.size(); i++) {
        cudaFree(z[i]);
        cudaFree(a[i]);
        cudaFree(deltas[i]);
    }

    return pos;
}