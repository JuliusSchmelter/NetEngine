#include <cassert>
#include <iostream>
#include <thread>

#include "Exceptions.h"
#include "Net.h"

//--------------------------------------------------------------------------------------------------
// Train the net.
void NetEngine::Net::train_on_cpu(const std::vector<std::vector<float>>& samples,
                                  const std::vector<std::vector<uint8_t>>& labels, size_t n_batches,
                                  size_t batch_size, size_t start_pos, size_t n_threads) {

    // std::thread::hardware_concurrency() can return 0
    if (n_threads == 0)
        n_threads = 4;

    std::cout << "training with " << n_threads << " threads ...\n";

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
    for (size_t i = 0; i < n_batches; i++) {
        std::cout << "batch " << i + 1 << ", pos = " << pos << '\n';

        // combine batches if next batch would not fit in training data
        size_t current_batch_size;
        if (pos + 2 * batch_size <= samples.size())
            current_batch_size = batch_size;
        else {
            current_batch_size = samples.size() - pos;
            i++;
        }

        // allocate storage for threads
        std::vector<std::vector<Eigen::MatrixXf>> weight_mods(n_threads);
        std::vector<std::vector<Eigen::Map<const Eigen::VectorXf>>> samples_eigen(n_threads);
        std::vector<std::vector<Eigen::Map<const Eigen::VectorX<uint8_t>>>> labels_eigen(n_threads);

        // split samples on threads
        size_t samples_per_thread = current_batch_size / n_threads;
        size_t remaining_samples = current_batch_size % n_threads;

        for (size_t j = 0; j < n_threads; j++) {
            // number of samples for current thread
            size_t n;
            if (j < remaining_samples)
                n = samples_per_thread + 1;
            else
                n = samples_per_thread;

            assert(pos + n <= samples.size());

            // get vectors of Eigen vectors for samples and labels, this
            // does not copy the data
            samples_eigen[j].reserve(n);
            labels_eigen[j].reserve(n);
            for (size_t k = 0; k < n; k++) {
                samples_eigen[j].push_back(Eigen::Map<const Eigen::VectorXf>(
                    samples[pos + k].data(), samples[pos + k].size()));

                labels_eigen[j].push_back(Eigen::Map<const Eigen::VectorX<uint8_t>>(
                    labels[pos + k].data(), labels[pos + k].size()));
            }

            // increment position in samples and labels
            pos += n;
        }

        // store threads
        std::vector<std::thread> threads;
        threads.reserve(n_threads);

        // dispatch threads
        for (size_t j = 0; j < n_threads; j++)
            threads.push_back(std::thread(&Net::get_weight_mods_cpu, this,
                                          std::ref(samples_eigen[j]), std::ref(labels_eigen[j]),
                                          std::ref(weight_mods[j])));

        // join threads and apply weight modifications
        // note: iterate backwards because later threads potentially have fewer
        // samples
        for (int j = n_threads - 1; j >= 0; j--) {
            threads[j].join();
            for (size_t k = 0; k < m_weights.size(); k++)
                m_weights[k] += weight_mods[j][k] / (float)current_batch_size;
        }

        // wrap around to beginning of training data if necessary
        if (pos == samples.size())
            pos = 0;
    }
    std::cout << "stopped training\n";
}

//--------------------------------------------------------------------------------------------------
// Calculate deltas for given samples and labels.
// Note: passing Eigen::Map instead of Eigen::Vector is necessary to preserve constness.
void NetEngine::Net::get_weight_mods_cpu(
    const std::vector<Eigen::Map<const Eigen::VectorXf>>& samples,
    const std::vector<Eigen::Map<const Eigen::VectorX<uint8_t>>>& labels,
    std::vector<Eigen::MatrixXf>& weight_mods) {
    // check for wrong dimensions
    if (samples.size() != labels.size())
        throw NetEngine::SetSizeError(samples.size(), labels.size());

    // init weight mods
    weight_mods.reserve(m_weights.size());
    for (size_t i = 0; i < m_weights.size(); i++)
        weight_mods.push_back(Eigen::MatrixXf::Zero(m_weights[i].rows(), m_weights[i].cols()));

    // temporary storage for activation values and deltas
    std::vector<Eigen::VectorXf> z(m_weights.size());
    std::vector<Eigen::VectorXf> a(m_weights.size());
    std::vector<Eigen::VectorXf> deltas;
    deltas.reserve(m_weights.size());
    for (size_t i = 0; i < m_weights.size(); i++)
        deltas.push_back(Eigen::VectorXf(m_layout[i + 1]));

    // run mini batch
    for (size_t i = 0; i < samples.size(); i++) {
        // run first layer
        // note: last column is bias
        z[0] = m_weights[0].leftCols(m_weights[0].cols() - 1) * samples[i] +
               m_weights[0].col(m_weights[0].cols() - 1);

        a[0] = z[0].unaryExpr([](float x) { return 0.5f + 0.5f * x / (1.0f + fabsf(x)); });

        // run remaining layers
        for (size_t j = 1; j < m_weights.size(); j++) {
            z[j] = m_weights[j].leftCols(m_weights[j].cols() - 1) * a[j - 1] +
                   m_weights[j].col(m_weights[j].cols() - 1);

            a[j] = z[j].unaryExpr([](float x) { return 0.5f + 0.5f * x / (1.0f + fabsf(x)); });
        }

        // get delta of last layer. get loss by comparing activation with
        // label, then take hadamard product with first derivative of
        // sigmoid of weighted input.
        for (size_t j = 0; j < m_layout.back(); j++)
            deltas.back()[j] = (a.back()[j] - labels[i][j]) / (1.0f + powf(fabsf(z.back()[j]), 2));

        // get delta of remaining layers. backpropagate delta of following
        // layer, then take hadamard product with first derivative of
        // sigmoid of weighted input
        for (size_t j = m_weights.size() - 1; j >= 1; j--)
            deltas[j - 1] = (m_weights[j].leftCols(m_weights[j].cols() - 1).transpose() * deltas[j])
                                .cwiseProduct(z[j - 1].unaryExpr(
                                    [](float x) { return 1.0f / (1.0f + powf(fabsf(x), 2)); }));

        // ouput weight modification
        // weights
        weight_mods[0].leftCols(weight_mods[0].cols() - 1) -=
            m_eta * deltas[0] * samples[i].transpose();

        // biases
        weight_mods[0].col(weight_mods[0].cols() - 1) -= m_eta_bias * deltas[0];

        for (size_t j = 1; j < weight_mods.size(); j++) {
            // weights
            weight_mods[j].leftCols(weight_mods[j].cols() - 1) -=
                m_eta * deltas[j] * a[j - 1].transpose();

            // biases
            weight_mods[j].col(weight_mods[j].cols() - 1) -= m_eta_bias * deltas[j];
        }
    }
}