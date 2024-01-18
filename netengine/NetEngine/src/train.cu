#include "DevTools.h"
#include "Exceptions.h"
#include "Net.h"

#include <cassert>
#include <iostream>

#define THREADS_PER_BLOCK 64

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

// GPU kernel for backpropagation.
__global__ void train_batch(size_t weight_layers, Eigen::Map<Eigen::MatrixXf>* weights,
                            Eigen::Map<Eigen::MatrixXf>* weight_mods,
                            Eigen::Map<Eigen::VectorXf>* a, Eigen::Map<Eigen::VectorXf>* z,
                            Eigen::Map<Eigen::VectorXf>* deltas, float** samples, uint32_t** labels,
                            size_t start_pos, size_t batch_size, float eta, float eta_bias) {
    // New batch, set weight mods to zero.
    for (size_t i = 0; i < weight_layers; i++) {
        weight_mods[i].setZero();
    }

    // Get weight mods for batch.
    for (size_t i = start_pos; i < start_pos + batch_size; i++) {
        // Wrap sample in Eigen vector.
        Eigen::Map<Eigen::VectorXf> sample_eigen(samples[i], weight_mods[0].cols() - 1);

        // Run first layer. Last column is bias.
        z[0] = weights[0].leftCols(weights[0].cols() - 1) * sample_eigen +
               weights[0].col(weights[0].cols() - 1);

        a[0] = z[0].unaryExpr([](float x) { return 0.5f + 0.5f * x / (1.0f + fabsf(x)); });

        // Run remaining layers.
        for (size_t j = 1; j < weight_layers; j++) {
            z[j] = weights[j].leftCols(weights[j].cols() - 1) * a[j - 1] +
                   weights[j].col(weights[j].cols() - 1);

            a[j] = z[j].unaryExpr([](float x) { return 0.5f + 0.5f * x / (1.0f + fabsf(x)); });
        }

        // Get delta of last layer. Get loss by comparing activation with label, then take
        // hadamard product with first derivative of sigmoid of weighted input.
        for (size_t j = 0; j < deltas[weight_layers - 1].size(); j++) {
            deltas[weight_layers - 1][j] = (a[weight_layers - 1][j] - labels[i][j]) /
                                           (1.0f + powf(fabsf(z[weight_layers - 1][j]), 2.0f));
        }

        // Get delta of remaining layers. Backpropagate delta of following layer, then take
        // hadamard product with first derivative of sigmoid of weighted input.
        for (size_t j = weight_layers - 1; j >= 1; j--) {
            deltas[j - 1] = (weights[j].leftCols(weights[j].cols() - 1).transpose() * deltas[j])
                                .cwiseProduct(z[j - 1].unaryExpr(
                                    [](float x) { return 1.0f / (1.0f + powf(fabsf(x), 2.0f)); }));
        }

        // Compute weight modification.
        weight_mods[0].leftCols(weight_mods[0].cols() - 1) -=
            eta * deltas[0] * sample_eigen.transpose();

        weight_mods[0].col(weight_mods[0].cols() - 1) -= eta_bias * deltas[0];

        for (size_t j = 1; j < weight_layers; j++) {
            weight_mods[j].leftCols(weight_mods[j].cols() - 1) -=
                eta * deltas[j] * a[j - 1].transpose();

            weight_mods[j].col(weight_mods[j].cols() - 1) -= eta_bias * deltas[j];
        }
    }

    // Apply weight mods.
    for (size_t i = 0; i < weight_layers; i++) {
        weights[i] += weight_mods[i] / (float)batch_size;
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