#include "netlib/exceptions.hpp"
#include "netlib/net.h"

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
    for (int i = 0; i < _weight_mods.size(); i++)
        _weight_mods[i] =
            Eigen::MatrixXf::Zero(m_weights[i].rows(), m_weights[i].cols());

    // temporary storage for activation values and deltas
    std::vector<Eigen::VectorXf> z(m_weights.size());
    std::vector<Eigen::VectorXf> a(m_weights.size());
    std::vector<Eigen::VectorXf> deltas;
    deltas.reserve(m_weights.size());
    for (int i = 0; i < m_weights.size(); i++)
        deltas.push_back(Eigen::VectorXf(m_layout[i + 1]));

    // run mini batch
    for (int i = 0; i < _samples.size(); i++)
    {
        // run first layer
        // note: last column is bias
        z[0] = m_weights[0].leftCols(m_weights[0].cols() - 1) * _samples[i] +
               m_weights[0].col(m_weights[0].cols() - 1);

        a[0] = z[0].unaryExpr([](float x)
                              { return 0.5f + 0.5f * x / (1.0f + fabsf(x)); });

        // run remaining layers
        for (int j = 1; j < m_weights.size(); j++)
        {
            z[j] = m_weights[j].leftCols(m_weights[j].cols() - 1) * a[j - 1] +
                   m_weights[j].col(m_weights[j].cols() - 1);

            a[j] = z[j].unaryExpr(
                [](float x) { return 0.5f + 0.5f * x / (1.0f + fabsf(x)); });
        }

        // get delta of last layer. get loss by comparing activation with
        // label, then take hadamard product with first derivative of
        // sigmoid of weighted input.
        for (int j = 0; j < m_layout.back(); j++)
            deltas.back()[j] = (a.back()[j] - _labels[i][j]) /
                               (1.0f + powf(fabsf(z.back()[j]), 2));

        // get delta of remaining layers. backpropagate delta of following
        // layer, then take hadamard product with first derivative of
        // sigmoid of weighted input
        for (int j = m_weights.size() - 1; j >= 1; j--)
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

        for (int j = 1; j < _weight_mods.size(); j++)
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
    std::vector<Eigen::MatrixXf> weight_mods(m_weights.size());

    // calculate weight mods
    get_weight_mods(sample, label, weight_mods);

    // apply weight mods
    for (int i = 0; i < m_weights.size(); i++)
        m_weights[i] += weight_mods[i];
}

//------------------------------------------------------------------------------
// training with mini batching and multithreading
//------------------------------------------------------------------------------
void netlib::net::train(const std::vector<std::vector<float>>& _samples,
                        const std::vector<std::vector<uint8_t>>& _labels,
                        size_t _batch_size, size_t _n_threads)
{
    // check for wrong dimensions
    if (_samples[0].size() != m_layout.front())
        throw netlib::dimension_error(_samples[0].size(), m_layout.front());

    if (_labels[0].size() != m_layout.back())
        throw netlib::dimension_error(_labels[0].size(), m_layout.back());
}
