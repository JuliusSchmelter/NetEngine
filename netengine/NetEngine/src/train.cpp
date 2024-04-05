#include <cassert>
#include <iostream>
#include <thread>

#include "Exceptions.h"
#include "Net.h"

// By combining multiple samples to mini batches, the training can be executed on multiple threads.
#define MINI_BATCH_SIZE 32

//--------------------------------------------------------------------------------------------------
// Train the net.
size_t NetEngine::Net::train_eigen(const std::vector<std::vector<float>>& samples,
                                   const std::vector<std::vector<uint8_t>>& labels,
                                   size_t n_samples, size_t start_pos) {
    // Check for bad inputs.
    if (samples[0].size() != m_layout.front())
        throw NetEngine::DimensionError(samples[0].size(), m_layout.front());

    if (labels[0].size() != m_layout.back())
        throw NetEngine::DimensionError(labels[0].size(), m_layout.back());

    if (samples.size() != labels.size())
        throw NetEngine::SetSizeError(samples.size(), labels.size());

    if (start_pos >= samples.size())
        start_pos = 0;

    // std::thread::hardware_concurrency() can return 0
    unsigned n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) {
        n_threads = 4;
        std::cerr
            << "Warning: std::thread::hardware_concurrency() returned 0, using 4 threads instead"
            << std::endl;
    } else if (n_threads > 32) {
        n_threads = 32;
    }

    // Samples per thread worker for one mini batch.
    size_t samples_per_thread = MINI_BATCH_SIZE / n_threads;

    // Allocate storage for threads.
    std::vector<std::vector<Eigen::MatrixXf>> weight_mods(n_threads);
    std::vector<std::vector<Eigen::VectorXf>> a(n_threads);
    std::vector<std::vector<Eigen::VectorXf>> z(n_threads);
    std::vector<std::vector<Eigen::VectorXf>> deltas(n_threads);

    for (size_t i = 0; i < n_threads; i++) {
        weight_mods[i].reserve(m_weights_eigen.size());
        a[i].reserve(m_weights_eigen.size());
        z[i].reserve(m_weights_eigen.size());
        deltas[i].reserve(m_weights_eigen.size());

        for (size_t j = 0; j < m_weights_eigen.size(); j++) {
            weight_mods[i].push_back(
                Eigen::MatrixXf(m_weights_eigen[j].rows(), m_weights_eigen[j].cols()));
            a[i].push_back(Eigen::VectorXf(m_layout[j + 1]));
            z[i].push_back(Eigen::VectorXf(m_layout[j + 1]));
            deltas[i].push_back(Eigen::VectorXf(m_layout[j + 1]));
        }
    }

    // Position in samples and labels.
    size_t pos = start_pos;

    // Counter of trained samples.
    size_t trained = 0;

    // Store threads.
    std::vector<std::thread> threads(n_threads);
    std::vector<size_t> samples_in_thread(n_threads);

    // Run mini batches.
    while (trained < n_samples) {
        for (size_t i = 0; i < n_threads; i++) {
            // Number of samples for current thread.
            if (trained + samples_per_thread <= n_samples)
                samples_in_thread[i] = samples_per_thread;
            else
                samples_in_thread[i] = n_samples - trained;

            trained += samples_in_thread[i];

            // Dispatch thread.
            threads[i] =
                std::thread(&Net::train_worker_eigen, this, std::ref(samples), std::ref(labels),
                            samples_in_thread[i], pos, std::ref(weight_mods[i]), std::ref(a[i]),
                            std::ref(z[i]), std::ref(deltas[i]));

            // Increment position in samples and labels.
            pos = (pos + samples_in_thread[i]) % samples.size();
        }

        // Join threads and apply weight modifications.
        // Iterate backwards because later threads potentially have fewer samples.
        for (int i = n_threads - 1; i >= 0; i--) {
            threads[i].join();
            for (size_t j = 0; j < m_weights_eigen.size(); j++)
                if (samples_in_thread[i] > 0)
                    m_weights_eigen[j] += weight_mods[i][j] / (float)samples_in_thread[i];
        }
    }

    return pos;
}

//--------------------------------------------------------------------------------------------------
// Calculate deltas for given samples and labels.
void NetEngine::Net::train_worker_eigen(const std::vector<std::vector<float>>& samples,
                                        const std::vector<std::vector<uint8_t>>& labels,
                                        size_t n_samples, size_t start_pos,
                                        std::vector<Eigen::MatrixXf>& weight_mods,
                                        std::vector<Eigen::VectorXf>& a,
                                        std::vector<Eigen::VectorXf>& z,
                                        std::vector<Eigen::VectorXf>& deltas) {
    // Check for wrong dimensions.
    if (samples.size() != labels.size())
        throw NetEngine::SetSizeError(samples.size(), labels.size());

    // Set weight modifications to zero.
    for (size_t i = 0; i < m_weights_eigen.size(); i++)
        weight_mods[i].setZero();

    // Run mini batch.
    size_t pos = start_pos;
    for (size_t i = 0; i < n_samples; i++) {
        // Get Eigen vectors for sample and label. This does not copy the data.
        Eigen::Map<const Eigen::VectorXf> sample(samples[pos].data(), samples[pos].size());
        Eigen::Map<const Eigen::VectorX<uint8_t>> label(labels[pos].data(), labels[pos].size());

        // Run first layer.
        // Note: last column is bias.
        z[0] = m_weights_eigen[0].leftCols(m_weights_eigen[0].cols() - 1) * sample +
               m_weights_eigen[0].col(m_weights_eigen[0].cols() - 1);

        a[0] = z[0].unaryExpr([](float x) { return 0.5f + 0.5f * x / (1.0f + fabsf(x)); });

        // Run remaining layers.
        for (size_t j = 1; j < m_weights_eigen.size(); j++) {
            z[j] = m_weights_eigen[j].leftCols(m_weights_eigen[j].cols() - 1) * a[j - 1] +
                   m_weights_eigen[j].col(m_weights_eigen[j].cols() - 1);

            a[j] = z[j].unaryExpr([](float x) { return 0.5f + 0.5f * x / (1.0f + fabsf(x)); });
        }

        // Get delta of last layer.
        // Get loss by comparing activation with label, then take hadamard product
        // with first derivative of sigmoid of weighted input.
        for (size_t j = 0; j < m_layout.back(); j++)
            deltas.back()[j] = (a.back()[j] - label[j]) / (1.0f + powf(fabsf(z.back()[j]), 2));

        // Get delta of remaining layers.
        // Backpropagate delta of following layer, then take hadamard product
        // with first derivative of sigmoid of weighted input.
        for (size_t j = m_weights_eigen.size() - 1; j >= 1; j--)
            deltas[j - 1] =
                (m_weights_eigen[j].leftCols(m_weights_eigen[j].cols() - 1).transpose() * deltas[j])
                    .cwiseProduct(z[j - 1].unaryExpr(
                        [](float x) { return 1.0f / (1.0f + powf(fabsf(x), 2)); }));

        // Output weight modification.
        // Weights.
        weight_mods[0].leftCols(weight_mods[0].cols() - 1) -=
            m_eta * deltas[0] * sample.transpose();

        // Biases.
        weight_mods[0].col(weight_mods[0].cols() - 1) -= m_eta_bias * deltas[0];

        for (size_t j = 1; j < weight_mods.size(); j++) {
            // Weights.
            weight_mods[j].leftCols(weight_mods[j].cols() - 1) -=
                m_eta * deltas[j] * a[j - 1].transpose();

            // Biases.
            weight_mods[j].col(weight_mods[j].cols() - 1) -= m_eta_bias * deltas[j];
        }

        // Update position in samples and labels.
        pos = (pos + 1) % samples.size();
    }
}
