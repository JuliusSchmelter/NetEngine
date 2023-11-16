#include <cassert>
#include <iostream>
#include <thread>

#include "Exceptions.h"
#include "Net.h"

// GPU kernel for backpropagation.
__global__ void backprop_sample(size_t weight_layers, Eigen::Map<Eigen::MatrixXf>* weights,
                                Eigen::Map<Eigen::MatrixXf>* weight_mods,
                                Eigen::Map<Eigen::VectorXf>* a, Eigen::Map<Eigen::VectorXf>* z,
                                Eigen::Map<Eigen::VectorXf>* deltas, float* sample, uint32_t* label,
                                float eta, float eta_bias) {
    Eigen::Map<Eigen::VectorXf> sample_eigen(sample, deltas[weight_layers - 1].size());

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
        deltas[weight_layers - 1][j] = (a[weight_layers - 1][j] - label[j]) /
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

    weight_mods[0].eval();

    for (size_t j = 1; j < weight_layers; j++) {
        weight_mods[j].leftCols(weight_mods[j].cols() - 1) -=
            eta * deltas[j] * a[j - 1].transpose();

        weight_mods[j].col(weight_mods[j].cols() - 1) -= eta_bias * deltas[j];

        weight_mods[j].eval();
    }
}

//--------------------------------------------------------------------------------------------------
// Train the net.
void NetEngine::Net::train(const std::vector<std::vector<float>>& samples,
                           const std::vector<std::vector<uint32_t>>& labels, size_t n_batches,
                           size_t batch_size, size_t start_pos, size_t n_threads) {
    // check for bad inputs
    if (samples[0].size() != m_layout.front())
        throw NetEngine::DimensionError(samples[0].size(), m_layout.front());

    if (labels[0].size() != m_layout.back())
        throw NetEngine::DimensionError(labels[0].size(), m_layout.back());

    if (samples.size() != labels.size())
        throw NetEngine::SetSizeError(samples.size(), labels.size());

    if (batch_size < n_threads)
        throw NetEngine::BatchesTooSmall(batch_size, n_threads);

    if (batch_size > samples.size())
        throw NetEngine::BatchesTooLarge(batch_size, samples.size());

    if (start_pos + batch_size > samples.size())
        start_pos = 0;

    // position in samples and labels
    size_t pos = start_pos;

    // batch loop
    for (size_t batch = 0; batch < n_batches; batch++) {
        std::cout << "batch " << batch + 1 << ", pos = " << pos << '\n';

        // combine batches if next batch would not fit in training data
        size_t current_batch_size;
        if (pos + 2 * batch_size <= samples.size())
            current_batch_size = batch_size;
        else {
            current_batch_size = samples.size() - pos;
            batch++;
        }

        // Init weight mods.
        std::vector<Eigen::MatrixXf> weight_mods;
        weight_mods.reserve(m_weights.size());
        for (size_t i = 0; i < m_weights.size(); i++)
            weight_mods.push_back(Eigen::MatrixXf::Zero(m_weights[i].rows(), m_weights[i].cols()));

        // Initialize weights and weight mods on the device.
        // Also, initialize temporary storage for activation values and deltas on device,
        // because allocating inside the kernel is not recommended.
        Eigen::Map<Eigen::MatrixXf>* device_weights;
        cudaMallocManaged(&device_weights, m_weights.size() * sizeof(Eigen::Map<Eigen::MatrixXf>));

        Eigen::Map<Eigen::MatrixXf>* device_weight_mods;
        cudaMallocManaged(&device_weight_mods,
                          m_weights.size() * sizeof(Eigen::Map<Eigen::MatrixXf>));

        Eigen::Map<Eigen::VectorXf>* device_a;
        cudaMallocManaged(&device_a, m_weights.size() * sizeof(Eigen::Map<Eigen::VectorXf>));

        Eigen::Map<Eigen::VectorXf>* device_z;
        cudaMallocManaged(&device_z, m_weights.size() * sizeof(Eigen::Map<Eigen::VectorXf>));

        Eigen::Map<Eigen::VectorXf>* device_deltas;
        cudaMallocManaged(&device_deltas, m_weights.size() * sizeof(Eigen::Map<Eigen::VectorXf>));

        for (size_t i = 0; i < m_weights.size(); i++) {
            float* data;
            cudaMalloc(&data, m_weights[i].size() * sizeof(float));
            cudaMemcpy(data, m_weights[i].data(), m_weights[i].size() * sizeof(float),
                       cudaMemcpyHostToDevice);
            device_weights[i] =
                Eigen::Map<Eigen::MatrixXf>(data, m_weights[i].rows(), m_weights[i].cols());

            float* zeros;
            cudaMalloc(&zeros, m_weights[i].size() * sizeof(float));
            cudaMemset(zeros, 0, m_weights[i].size() * sizeof(float));
            device_weight_mods[i] =
                Eigen::Map<Eigen::MatrixXf>(zeros, m_weights[i].rows(), m_weights[i].cols());

            float* a;
            cudaMalloc(&a, m_layout[i + 1] * sizeof(float));
            device_a[i] = Eigen::Map<Eigen::VectorXf>(a, m_layout[i + 1]);

            float* z;
            cudaMalloc(&z, m_layout[i + 1] * sizeof(float));
            device_z[i] = Eigen::Map<Eigen::VectorXf>(z, m_layout[i + 1]);

            float* deltas;
            cudaMalloc(&deltas, m_layout[i + 1] * sizeof(float));
            device_deltas[i] = Eigen::Map<Eigen::VectorXf>(deltas, m_layout[i + 1]);
        }

        // Run batch.
        for (size_t cursor = pos; cursor < pos + current_batch_size; cursor++) {
            // Allocate sample and label on device.
            float* sample;
            uint32_t* label;

            cudaMalloc(&sample, samples[cursor].size() * sizeof(float));
            cudaMalloc(&label, labels[cursor].size() * sizeof(uint32_t));

            cudaMemcpy(sample, samples[cursor].data(), samples[cursor].size() * sizeof(float),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(label, labels[cursor].data(), labels[cursor].size() * sizeof(uint32_t),
                       cudaMemcpyHostToDevice);

            // Run backprop kernel.
            backprop_sample<<<1, 1>>>(m_weights.size(), device_weights, device_weight_mods,
                                      device_a, device_z, device_deltas, sample, label, m_eta,
                                      m_eta_bias);
            cudaDeviceSynchronize();

            cudaFree((void*)sample);
            cudaFree((void*)label);
        }

        // Retrieve weight mods from device.
        for (size_t i = 0; i < m_weights.size(); i++) {
            cudaMemcpy(weight_mods[i].data(), &device_weight_mods[i],
                       weight_mods[i].size() * sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree((void*)&device_weights[i]);
            cudaFree((void*)&device_weight_mods[i]);
            cudaFree((void*)&device_a[i]);
            cudaFree((void*)&device_z[i]);
            cudaFree((void*)&device_deltas[i]);
        }

        cudaFree((void*)device_weights);
        cudaFree((void*)device_weight_mods);
        cudaFree((void*)device_a);
        cudaFree((void*)device_z);
        cudaFree((void*)device_deltas);

        for (size_t i = 0; i < m_weights.size(); i++)
            m_weights[i] += weight_mods[i] / (float)current_batch_size;

        pos += current_batch_size;

        // wrap around to beginning of training data if necessary
        if (pos == samples.size())
            pos = 0;
    }

    std::cout << "stopped training\n";
}