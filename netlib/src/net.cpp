#include "netlib/net.h"
#include "netlib/exceptions.hpp"

#include <cassert>
#include <iomanip>
#include <iostream>

netlib::net::net(const std::vector<int>& _layout, float _eta, float _eta_bias)
    : m_layout(_layout), m_eta(_eta), m_eta_bias(_eta_bias)
{
    // initialize weight matrices, add column for bias
    for (int i = 0; i < _layout.size() - 1; i++)
        m_weights.push_back(Eigen::MatrixXf(_layout[i + 1], _layout[i] + 1));
}

netlib::net::net(const std::vector<int>& _layout, float _eta)
    : net(_layout, _eta, 0.2 * _eta)
{
}

float netlib::net::get_eta()
{
    return m_eta;
}

void netlib::net::set_eta(float _eta)
{
    m_eta = _eta;
}

float netlib::net::get_eta_bias()
{
    return m_eta_bias;
}

void netlib::net::set_eta_bias(float _eta_bias)
{
    m_eta_bias = _eta_bias;
}

void netlib::net::print()
{
    // print layout
    std::cout << "##############################################\nlayout: |";

    for (auto& i : m_layout)
        std::cout << i << "|";

    // print eta and threshold
    std::cout << "\neta: " << m_eta << "\neta_bias: " << m_eta_bias;

    // print weights
    std::cout << " \nweights : ";

    for (auto& i : m_weights)
        std::cout << "\n----------------------------------------------\n"
                  << std::fixed << std::setprecision(3) << i;

    std::cout << "\n----------------------------------------------\n";
}

void netlib::net::set_random()
{
    // seed random number generator to get new random numbers each time
    std::srand((unsigned int)std::time(0));

    // set weight matrices to random values in range [-1, 1]
    for (auto& i : m_weights)
        i.setRandom();
}

std::vector<float> netlib::net::run(const std::vector<float>& _input)
{
    // check dimensions
    if (_input.size() != m_layout.front())
        throw netlib::dimension_error(_input.size(), m_layout.front());

    // get Eigen vector for input, this does not copy the data
    Eigen::Map<const Eigen::VectorXf> input(_input.data(), _input.size());

    // run first layer
    // note: last column is bias
    Eigen::VectorXf a =
        (m_weights[0].leftCols(m_weights[0].cols() - 1) * input +
         m_weights[0].col(m_weights[0].cols() - 1))
            .unaryExpr([](float x) { return (x > 0.0f) ? x : 0.0f; });

    // run remaining layers
    for (int i = 1; i < m_weights.size(); i++)
    {
        // note: eval() forces evaluation before new values are assigned to a
        a = ((m_weights[i].leftCols(m_weights[i].cols() - 1) * a).eval() +
             m_weights[i].col(m_weights[i].cols() - 1))
                .unaryExpr([](float x) { return (x > 0.0f) ? x : 0.0f; });
    }

    return std::vector<float>(a.data(), a.data() + a.size());
}

void netlib::net::train(const std::vector<float>& _input,
                        const std::vector<uint8_t>& _label)
{
    // check for wrong dimensions
    if (_input.size() != m_layout.front())
        throw netlib::dimension_error(_input.size(), m_layout.front());

    if (_label.size() != m_layout.back())
        throw netlib::dimension_error(_label.size(), m_layout.back());

    // get Eigen vector for input, this does not copy the data
    Eigen::Map<const Eigen::VectorXf> input(_input.data(), _input.size());

    // store activation values
    std::vector<Eigen::VectorXf> z(m_weights.size());
    std::vector<Eigen::VectorXf> a(m_weights.size());

    // run first layer
    // note: last column is bias
    z[0] = m_weights[0].leftCols(m_weights[0].cols() - 1) * input +
           m_weights[0].col(m_weights[0].cols() - 1);

    a[0] = z[0].unaryExpr([](float x) { return (x > 0.0f) ? x : 0.0f; });

    // run remaining layers
    for (int i = 1; i < m_weights.size(); i++)
    {
        z[i] = m_weights[i].leftCols(m_weights[i].cols() - 1) * a[i - 1] +
               m_weights[i].col(m_weights[i].cols() - 1);

        a[i] = z[i].unaryExpr([](float x) { return (x > 0.0f) ? x : 0.0f; });
    }

    // get storage for delta of each layer
    std::vector<Eigen::VectorXf> delta;

    for (int i = 1; i < m_layout.size(); i++)
        delta.push_back(Eigen::VectorXf(m_layout[i]));

    // get delta of last layer. get loss by comparing activation with
    // threshold and label, then take hadamard product with first derivative
    // of ReLU of weighted input
    for (int i = 0; i < m_layout.back(); i++)
        delta.back()[i] =
            (a.back()[i] - _label[i]) * (z.back()[i] > 0.0f ? 1.0f : 0.0f);

    // get delta of other layers. backpropagate delta of following layer, then
    // take hadamard product with first derivative of ReLU of weighted input
    for (int i = m_weights.size() - 1; i >= 1; i--)
        delta[i - 1] =
            (m_weights[i].leftCols(m_weights[i].cols() - 1).transpose() *
             delta[i])
                .cwiseProduct(z[i - 1].unaryExpr(
                    [](float x) { return (x > 0.0f) ? 1.0f : 0.0f; }));

    // apply delta to weights and biases
    // weights
    m_weights[0].leftCols(m_weights[0].cols() - 1) -=
        m_eta * delta[0] * input.transpose();

    // biases
    m_weights[0].col(m_weights[0].cols() - 1) -= m_eta_bias * delta[0];

    for (int i = 1; i < m_weights.size(); i++)
    {
        // weights
        m_weights[i].leftCols(m_weights[i].cols() - 1) -=
            m_eta * delta[i] * a[i - 1].transpose();

        // biases
        m_weights[i].col(m_weights[i].cols() - 1) -= m_eta_bias * delta[i];
    }
}

float netlib::net::test(const std::vector<std::vector<float>>& _samples,
                        const std::vector<uint8_t>& _labels)
{
    if (_samples.size() != _labels.size())
        throw netlib::set_size_error(_samples.size(), _labels.size());

    unsigned success = 0;

    for (int i = 0; i < _samples.size(); i++)
    {
        std::vector<float> output = run(_samples[i]);

        uint8_t result =
            std::max_element(output.begin(), output.end()) - output.begin();

        if (result == _labels[i])
            success++;
    }

    return success / _samples.size();
}