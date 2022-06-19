#include "netlib/exceptions.hpp"
#include "netlib/net.h"

#include <cassert>
#include <iostream>
#include <thread>

//------------------------------------------------------------------------------
// calculate deltas for given samples and labels
//------------------------------------------------------------------------------
// note: passing Eigen::Map instead of Eigen::Vector is necessary to preserve
// constness
void netlib::net::get_weight_mods(
    const std::vector<Eigen::Map<const Eigen::VectorXf>>& _samples,
    const std::vector<Eigen::Map<const Eigen::VectorX<uint8_t>>>& _labels,
    std::vector<Eigen::MatrixXf>& _weight_mods)
{
    // check for wrong dimensions
    if (_samples.size() != _labels.size())
        throw netlib::set_size_error(_samples.size(), _labels.size());

    // init weight mods
    _weight_mods.reserve(m_weights.size());
    for (size_t i = 0; i < m_weights.size(); i++)
        _weight_mods.push_back(
            Eigen::MatrixXf::Zero(m_weights[i].rows(), m_weights[i].cols()));

    // temporary storage for activation values and deltas
    std::vector<Eigen::VectorXf> z(m_weights.size());
    std::vector<Eigen::VectorXf> a(m_weights.size());
    std::vector<Eigen::VectorXf> deltas;
    deltas.reserve(m_weights.size());
    for (size_t i = 0; i < m_weights.size(); i++)
        deltas.push_back(Eigen::VectorXf(m_layout[i + 1]));

    // run mini batch
    for (size_t i = 0; i < _samples.size(); i++)
    {
        // run first layer
        // note: last column is bias
        z[0] = m_weights[0].leftCols(m_weights[0].cols() - 1) * _samples[i] +
               m_weights[0].col(m_weights[0].cols() - 1);

        a[0] = z[0].unaryExpr([](float x)
                              { return 0.5f + 0.5f * x / (1.0f + fabsf(x)); });

        // run remaining layers
        for (size_t j = 1; j < m_weights.size(); j++)
        {
            z[j] = m_weights[j].leftCols(m_weights[j].cols() - 1) * a[j - 1] +
                   m_weights[j].col(m_weights[j].cols() - 1);

            a[j] = z[j].unaryExpr(
                [](float x) { return 0.5f + 0.5f * x / (1.0f + fabsf(x)); });
        }

        // get delta of last layer. get loss by comparing activation with
        // label, then take hadamard product with first derivative of
        // sigmoid of weighted input.
        for (size_t j = 0; j < m_layout.back(); j++)
            deltas.back()[j] = (a.back()[j] - _labels[i][j]) /
                               (1.0f + powf(fabsf(z.back()[j]), 2));

        // get delta of remaining layers. backpropagate delta of following
        // layer, then take hadamard product with first derivative of
        // sigmoid of weighted input
        for (size_t j = m_weights.size() - 1; j >= 1; j--)
            deltas[j - 1] =
                (m_weights[j].leftCols(m_weights[j].cols() - 1).transpose() *
                 deltas[j])
                    .cwiseProduct(z[j - 1].unaryExpr(
                        [](float x)
                        { return 1.0f / (1.0f + powf(fabsf(x), 2)); }));

        // ouput weight modification
        // weights
        _weight_mods[0].leftCols(_weight_mods[0].cols() - 1) -=
            m_eta * deltas[0] * _samples[i].transpose();

        // biases
        _weight_mods[0].col(_weight_mods[0].cols() - 1) -=
            m_eta_bias * deltas[0];

        for (size_t j = 1; j < _weight_mods.size(); j++)
        {
            // weights
            _weight_mods[j].leftCols(_weight_mods[j].cols() - 1) -=
                m_eta * deltas[j] * a[j - 1].transpose();

            // biases
            _weight_mods[j].col(_weight_mods[j].cols() - 1) -=
                m_eta_bias * deltas[j];
        }
    }
}

//------------------------------------------------------------------------------
// basic training, no mini batching or multithreading
//------------------------------------------------------------------------------
void netlib::net::train(const std::vector<float>& _sample,
                        const std::vector<uint8_t>& _label)
{
    // check for wrong dimensions
    if (_sample.size() != m_layout.front())
        throw netlib::dimension_error(_sample.size(), m_layout.front());

    if (_label.size() != m_layout.back())
        throw netlib::dimension_error(_label.size(), m_layout.back());

    // get vector of Eigen vector for sample, this does not copy the data
    std::vector<Eigen::Map<const Eigen::VectorXf>> sample;
    sample.push_back(
        Eigen::Map<const Eigen::VectorXf>(_sample.data(), _sample.size()));

    // get vector of Eigen vector for label, this does not copy the data
    std::vector<Eigen::Map<const Eigen::VectorX<uint8_t>>> label;
    label.push_back(Eigen::Map<const Eigen::VectorX<uint8_t>>(_label.data(),
                                                              _label.size()));

    // get vector for weight mods
    std::vector<Eigen::MatrixXf> weight_mods;

    // calculate weight mods
    get_weight_mods(sample, label, weight_mods);

    // apply weight mods
    for (size_t i = 0; i < m_weights.size(); i++)
        m_weights[i] += weight_mods[i];
}

//------------------------------------------------------------------------------
// training with mini batching and multithreading
//------------------------------------------------------------------------------
void netlib::net::train(const std::vector<std::vector<float>>& _samples,
                        const std::vector<std::vector<uint8_t>>& _labels,
                        size_t _n_batches, size_t _batch_size,
                        size_t _n_threads, size_t _start_pos)
{
    std::cout << "training...\n";

    // check for bad inputs
    if (_samples[0].size() != m_layout.front())
        throw netlib::dimension_error(_samples[0].size(), m_layout.front());

    if (_labels[0].size() != m_layout.back())
        throw netlib::dimension_error(_labels[0].size(), m_layout.back());

    if (_samples.size() != _labels.size())
        throw netlib::set_size_error(_samples.size(), _labels.size());

    if (_batch_size < _n_threads)
        throw netlib::batches_too_small(_batch_size, _n_threads);

    if (_batch_size > _samples.size())
        throw netlib::batches_too_large(_batch_size, _samples.size());

    if (_start_pos + _batch_size > _samples.size())
        throw netlib::invalid_start_pos(_start_pos);

    // position in _samples and _labels
    size_t pos = _start_pos;

    // batch loop
    for (size_t i = 0; i < _n_batches; i++)
    {
        std::cout << "batch " << i + 1 << ", pos = " << pos << '\n';

        // combine batches if next batch would not fit in training data
        size_t batch_size;
        if (pos + 2 * _batch_size <= _samples.size())
            batch_size = _batch_size;
        else
        {
            batch_size = _samples.size() - pos;
            i++;
        }

        // allocate storage for threads
        std::vector<std::vector<Eigen::MatrixXf>> weight_mods(_n_threads);
        std::vector<std::vector<Eigen::Map<const Eigen::VectorXf>>> samples(
            _n_threads);
        std::vector<std::vector<Eigen::Map<const Eigen::VectorX<uint8_t>>>>
            labels(_n_threads);

        // split samples on threads
        size_t samples_per_thread = batch_size / _n_threads;
        size_t remaining_samples = batch_size % _n_threads;

        for (size_t j = 0; j < _n_threads; j++)
        {
            // number of samples for current thread
            size_t n;
            if (j < remaining_samples)
                n = samples_per_thread + 1;
            else
                n = samples_per_thread;

            // avoid segfaults
            assert(pos + n <= _samples.size());

            // get vectors of Eigen vectors for samples and labels, this
            // does not copy the data
            samples[j].reserve(n);
            labels[j].reserve(n);
            for (size_t k = 0; k < n; k++)
            {
                samples[j].push_back(Eigen::Map<const Eigen::VectorXf>(
                    _samples[pos + k].data(), _samples[pos + k].size()));

                labels[j].push_back(Eigen::Map<const Eigen::VectorX<uint8_t>>(
                    _labels[pos + k].data(), _labels[pos + k].size()));
            }

            // increment position in _samples and _labels
            pos += n;
        }

        // store threads
        std::vector<std::thread> threads;
        threads.reserve(_n_threads);

        // dispatch threads
        for (size_t j = 0; j < _n_threads; j++)
            threads.push_back(
                std::thread(&net::get_weight_mods, this, std::ref(samples[j]),
                            std::ref(labels[j]), std::ref(weight_mods[j])));

        // join threads and apply weight modifications
        // note: iterate backwards because later threads potentially have fever
        // samples
        for (size_t j = _n_threads - 1; j >= 0; j--)
        {
            threads[j].join();
            for (size_t k = 0; k < m_weights.size(); k++)
                m_weights[k] += weight_mods[j][k] / (float)batch_size;
        }

        // wrap around to beginning of training data if necessary
        if (pos == _samples.size())
            pos = 0;
    }
    std::cout << "stopped training\n";
}
