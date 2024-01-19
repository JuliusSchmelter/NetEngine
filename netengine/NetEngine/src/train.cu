#include <cassert>
#include <iostream>

#include "DevTools.h"
#include "Exceptions.h"
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

void NetEngine::test_func() {
    std::cout << "test_func\n";

    float A[6]{1, 2, 3, 4, 5, 6};
    uint32_t A_rows = 2;
    uint32_t A_cols = 3;

    float B[6]{7, 8, 9, 10, 11, 12};
    uint32_t B_rows = 3;
    uint32_t B_cols = 2;

    float C[4];
    uint32_t C_rows = 2;
    uint32_t C_cols = 2;

    // Compute C = A * B on GPU.
    float* device_A;
    TRY_CUDA(cudaMalloc(&device_A, A_rows * A_cols * sizeof(float)));
    TRY_CUDA(cudaMemcpy(device_A, A, A_rows * A_cols * sizeof(float), cudaMemcpyHostToDevice));

    float* device_B;
    TRY_CUDA(cudaMalloc(&device_B, B_rows * B_cols * sizeof(float)));
    TRY_CUDA(cudaMemcpy(device_B, B, B_rows * B_cols * sizeof(float), cudaMemcpyHostToDevice));

    float* device_C;
    TRY_CUDA(cudaMalloc(&device_C, C_rows * C_cols * sizeof(float)));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((C_cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (C_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul<<<numBlocks, threadsPerBlock>>>(device_A, device_B, device_C, A_rows, A_cols, B_cols);

    TRY_CUDA(cudaMemcpy(C, device_C, C_rows * C_cols * sizeof(float), cudaMemcpyDeviceToHost));
    TRY_CUDA(cudaFree(device_A));
    TRY_CUDA(cudaFree(device_B));
    TRY_CUDA(cudaFree(device_C));

    // Print result.
    for (size_t i = 0; i < C_rows; i++) {
        for (size_t j = 0; j < C_cols; j++) {
            std::cout << C[i * C_cols + j] << ' ';
        }
        std::cout << '\n';
    }
}

//--------------------------------------------------------------------------------------------------
// Train the net.
void NetEngine::Net::train(const std::vector<std::vector<float>>& samples,
                           const std::vector<std::vector<uint32_t>>& labels, size_t n_batches,
                           size_t batch_size, size_t start_pos) {
    // check for bad inputs
    if (samples[0].size() != m_layout.front())
        throw NetEngine::DimensionError(samples[0].size(), m_layout.front());

    if (labels[0].size() != m_layout.back())
        throw NetEngine::DimensionError(labels[0].size(), m_layout.back());

    if (samples.size() != labels.size())
        throw NetEngine::SetSizeError(samples.size(), labels.size());

    if (batch_size > samples.size())
        throw NetEngine::BatchesTooLarge(batch_size, samples.size());

    if (start_pos + batch_size > samples.size())
        start_pos = 0;

    // Position in samples and labels.
    size_t pos = start_pos;

    // Copy samples, labels and weights to device.
    float** device_samples;
    TRY_CUDA(cudaMallocManaged(&device_samples, samples.size() * sizeof(float*)));

    uint32_t** device_labels;
    TRY_CUDA(cudaMallocManaged(&device_labels, labels.size() * sizeof(uint32_t*)));

    for (size_t i = 0; i < samples.size(); i++) {
        float* sample;
        TRY_CUDA(cudaMalloc(&sample, samples[i].size() * sizeof(float)));
        TRY_CUDA(cudaMemcpy(sample, samples[i].data(), samples[i].size() * sizeof(float),
                            cudaMemcpyHostToDevice));
        device_samples[i] = sample;

        uint32_t* label;
        TRY_CUDA(cudaMalloc(&label, labels[i].size() * sizeof(uint32_t)));
        TRY_CUDA(cudaMemcpy(label, labels[i].data(), labels[i].size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice));
        device_labels[i] = label;
    }

    Eigen::Map<Eigen::MatrixXf>* device_weights;
    TRY_CUDA(
        cudaMallocManaged(&device_weights, m_weights.size() * sizeof(Eigen::Map<Eigen::MatrixXf>)));

    for (size_t i = 0; i < m_weights.size(); i++) {
        float* weights;
        TRY_CUDA(cudaMalloc(&weights, m_weights[i].size() * sizeof(float)));
        TRY_CUDA(cudaMemcpy(weights, m_weights[i].data(), m_weights[i].size() * sizeof(float),
                            cudaMemcpyHostToDevice));
        new (&device_weights[i])
            Eigen::Map<Eigen::MatrixXf>(weights, m_weights[i].rows(), m_weights[i].cols());
    }

    // Determine CUDA grid dimensions.
    // Get number of multiprocessors, use it as number of blocks.
    int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    int n_blocks = props.multiProcessorCount;
    int n_threads = n_blocks * THREADS_PER_BLOCK;
    int samples_per_thread = batch_size / n_threads ? batch_size / n_threads : 1;
    int real_batch_size = n_threads * samples_per_thread;
    LOG(real_batch_size)

    // Initialize temporary storage on device, because allocating inside the kernel
    // is not recommended.

    // Each block needs a weight_mods instance.
    Eigen::Map<Eigen::MatrixXf>* device_weight_mods;
    TRY_CUDA(cudaMallocManaged(&device_weight_mods,
                               m_weights.size() * sizeof(Eigen::Map<Eigen::MatrixXf>)));

    Eigen::Map<Eigen::VectorXf>* device_a;
    TRY_CUDA(cudaMallocManaged(&device_a, m_weights.size() * sizeof(Eigen::Map<Eigen::VectorXf>)));

    Eigen::Map<Eigen::VectorXf>* device_z;
    TRY_CUDA(cudaMallocManaged(&device_z, m_weights.size() * sizeof(Eigen::Map<Eigen::VectorXf>)));

    Eigen::Map<Eigen::VectorXf>* device_deltas;
    TRY_CUDA(
        cudaMallocManaged(&device_deltas, m_weights.size() * sizeof(Eigen::Map<Eigen::VectorXf>)));

    for (size_t i = 0; i < m_weights.size(); i++) {
        float* weights;
        TRY_CUDA(cudaMalloc(&weights, m_weights[i].size() * sizeof(float)));
        TRY_CUDA(cudaMemcpy(weights, m_weights[i].data(), m_weights[i].size() * sizeof(float),
                            cudaMemcpyHostToDevice));
        new (&device_weights[i])
            Eigen::Map<Eigen::MatrixXf>(weights, m_weights[i].rows(), m_weights[i].cols());

        float* mods;
        TRY_CUDA(cudaMalloc(&mods, m_weights[i].size() * sizeof(float)));
        new (&device_weight_mods[i])
            Eigen::Map<Eigen::MatrixXf>(mods, m_weights[i].rows(), m_weights[i].cols());

        float* a;
        TRY_CUDA(cudaMalloc(&a, m_layout[i + 1] * sizeof(float)));
        new (&device_a[i]) Eigen::Map<Eigen::VectorXf>(a, m_layout[i + 1]);

        float* z;
        TRY_CUDA(cudaMalloc(&z, m_layout[i + 1] * sizeof(float)));
        new (&device_z[i]) Eigen::Map<Eigen::VectorXf>(z, m_layout[i + 1]);

        float* deltas;
        TRY_CUDA(cudaMalloc(&deltas, m_layout[i + 1] * sizeof(float)));
        new (&device_deltas[i]) Eigen::Map<Eigen::VectorXf>(deltas, m_layout[i + 1]);
    }

    // Train net in batches.
    for (size_t batch = 0; batch < n_batches; batch++) {
        std::cout << "batch " << batch + 1 << ", pos = " << pos << '\n';

        // Train batch on GPU.
        train_batch<<<n_blocks, THREADS_PER_BLOCK>>>(
            m_weights.size(), device_weights, device_weight_mods, device_a, device_z, device_deltas,
            device_samples, device_labels, pos, real_batch_size, m_eta, m_eta_bias);

        TRY_CUDA(cudaDeviceSynchronize());

        pos += real_batch_size;

        // Wrap around to beginning of training data if necessary.
        if (pos == samples.size())
            pos = 0;
    }

    // Retrieve weights from device and free memory.
    for (size_t i = 0; i < m_weights.size(); i++) {
        TRY_CUDA(cudaMemcpy(m_weights[i].data(), device_weights[i].data(),
                            m_weights[i].size() * sizeof(float), cudaMemcpyDeviceToHost));

        TRY_CUDA(cudaDeviceSynchronize());

        TRY_CUDA(cudaFree(device_weights[i].data()));
        TRY_CUDA(cudaFree(device_weight_mods[i].data()));
        TRY_CUDA(cudaFree(device_a[i].data()));
        TRY_CUDA(cudaFree(device_z[i].data()));
        TRY_CUDA(cudaFree(device_deltas[i].data()));
    }

    for (size_t i = 0; i < samples.size(); i++) {
        TRY_CUDA(cudaFree(device_samples[i]));
        TRY_CUDA(cudaFree(device_labels[i]));
    }

    TRY_CUDA(cudaFree(device_samples));
    TRY_CUDA(cudaFree(device_labels));
    TRY_CUDA(cudaFree(device_weights));
    TRY_CUDA(cudaFree(device_a));
    TRY_CUDA(cudaFree(device_z));
    TRY_CUDA(cudaFree(device_deltas));
    TRY_CUDA(cudaFree(device_weight_mods));
}