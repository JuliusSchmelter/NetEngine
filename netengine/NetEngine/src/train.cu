#include <cassert>
#include <iostream>
#include <thread>

#include "Exceptions.h"
#include "Net.h"

//--------------------------------------------------------------------------------------------------
// Train the net.
void NetEngine::Net::train(const std::vector<std::vector<float>>& samples,
                           const std::vector<std::vector<uint8_t>>& labels, size_t n_batches,
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

        // Wrap samples and labels for eigen.
        std::vector<Eigen::Map<const Eigen::VectorXf>> batch_samples;
        batch_samples.reserve(current_batch_size);

        std::vector<Eigen::Map<const Eigen::VectorX<uint8_t>>> batch_labels;
        batch_labels.reserve(current_batch_size);

        for (size_t cursor = pos; cursor < pos + current_batch_size; cursor++) {
            batch_samples.push_back(
                Eigen::Map<const Eigen::VectorXf>(samples[cursor].data(), samples[cursor].size()));

            batch_labels.push_back(Eigen::Map<const Eigen::VectorX<uint8_t>>(
                labels[cursor].data(), labels[cursor].size()));
        }

        pos += current_batch_size;

        // Init weight mods.
        std::vector<Eigen::MatrixXf> weight_mods;
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

        // run batch
        for (size_t sample = 0; sample < batch_samples.size(); sample++) {
            // run first layer
            // note: last column is bias
            z[0] = m_weights[0].leftCols(m_weights[0].cols() - 1) * batch_samples[sample] +
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
                deltas.back()[j] =
                    (a.back()[j] - batch_labels[sample][j]) / (1.0f + powf(fabsf(z.back()[j]), 2));

            // get delta of remaining layers. backpropagate delta of following
            // layer, then take hadamard product with first derivative of
            // sigmoid of weighted input
            for (size_t j = m_weights.size() - 1; j >= 1; j--)
                deltas[j - 1] =
                    (m_weights[j].leftCols(m_weights[j].cols() - 1).transpose() * deltas[j])
                        .cwiseProduct(z[j - 1].unaryExpr(
                            [](float x) { return 1.0f / (1.0f + powf(fabsf(x), 2)); }));

            // Output weight modification.
            // weights
            weight_mods[0].leftCols(weight_mods[0].cols() - 1) -=
                m_eta * deltas[0] * batch_samples[sample].transpose();

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

        for (size_t i = 0; i < m_weights.size(); i++)
            m_weights[i] += weight_mods[i] / (float)current_batch_size;

        // wrap around to beginning of training data if necessary
        if (pos == samples.size())
            pos = 0;
    }

    std::cout << "stopped training\n";
}