#include "netlib/net.h"
#include "netlib/exceptions.hpp"

#include <cassert>
#include <iomanip>
#include <iostream>
#include <math.h>

//------------------------------------------------------------------------------
// constructor
//------------------------------------------------------------------------------
netlib::net::net(const std::vector<unsigned>& _layout, float _eta, float _eta_bias)
    : m_layout(_layout), m_eta(_eta), m_eta_bias(_eta_bias)
{
    // check min number of layers
    if(_layout.size() < 3)
        throw netlib::not_enough_layers(_layout.size());
        
    // initialize weight matrices, add column for bias
    for (int i = 0; i < _layout.size() - 1; i++)
        m_weights.push_back(Eigen::MatrixXf(_layout[i + 1], _layout[i] + 1));
}

netlib::net::net(const std::vector<unsigned>& _layout, float _eta)
    : net(_layout, _eta, 0.2 * _eta)
{
}

//------------------------------------------------------------------------------
// get, set
//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
// print net
//------------------------------------------------------------------------------
void netlib::net::print()
{
    // save stream format
    std::ios streamfmt(nullptr);
    streamfmt.copyfmt(std::cout);

    // print layout
    std::cout << "##############################################\nlayout: |";

    for (auto& i : m_layout)
        std::cout << i << "|";

    // print eta and threshold
    std::cout << "\neta: " << m_eta << "\neta_bias: " << m_eta_bias
              << "\nn_parameters: " << n_parameters();

    // print weights
    std::cout << " \nweights : ";

    for (auto& i : m_weights)
        std::cout << "\n----------------------------------------------\n"
                  << std::fixed << std::setprecision(3) << i;

    std::cout << "\n----------------------------------------------\n";

    // restore stream format
    std::cout.copyfmt(streamfmt);
}

//------------------------------------------------------------------------------
// get number of parameters
//------------------------------------------------------------------------------
size_t netlib::net::n_parameters()
{
    size_t n = 0;
    for (const auto& i : m_weights)
        n += i.rows() * i.cols();

    return n;
}

//------------------------------------------------------------------------------
// set weights to random values
//------------------------------------------------------------------------------
void netlib::net::set_random()
{
    // seed random number generator to get new random numbers each time
    std::srand((unsigned int)std::time(0));

    for (auto& i : m_weights)
    {
        // set weight matrix to random values in range [-1, 1]
        i.setRandom();

        // change range to [+- 1/sqrt(number of neurons in layer)]
        i / sqrt(i.cols() - 1);
    }
}

//------------------------------------------------------------------------------
// run example
//------------------------------------------------------------------------------
std::vector<float> netlib::net::run(const std::vector<float>& _sample)
{
    // check dimensions
    if (_sample.size() != m_layout.front())
        throw netlib::dimension_error(_sample.size(), m_layout.front());

    // get Eigen vector for input, this does not copy the data
    Eigen::Map<const Eigen::VectorXf> input(_sample.data(), _sample.size());

    // run first layer
    // note: last column is bias
    Eigen::VectorXf a =
        (m_weights[0].leftCols(m_weights[0].cols() - 1) * input +
         m_weights[0].col(m_weights[0].cols() - 1))
            .unaryExpr([](float x)
                       { return 0.5f + 0.5f * x / (1.0f + fabsf(x)); });

    // run remaining layers
    for (int i = 1; i < m_weights.size(); i++)
    {
        // note: eval() forces evaluation before new values are assigned to a
        a = ((m_weights[i].leftCols(m_weights[i].cols() - 1) * a).eval() +
             m_weights[i].col(m_weights[i].cols() - 1))
                .unaryExpr([](float x)
                           { return 0.5f + 0.5f * x / (1.0f + fabsf(x)); });
    }

    return std::vector<float>(a.data(), a.data() + a.size());
}

//------------------------------------------------------------------------------
// test accuracy
//------------------------------------------------------------------------------
float netlib::net::test(const std::vector<std::vector<float>>& _samples,
                        const std::vector<std::vector<uint8_t>>& _labels,
                        float _threshold)
{
    if (_samples.size() != _labels.size())
        throw netlib::set_size_error(_samples.size(), _labels.size());

    unsigned success = 0;

    for (int i = 0; i < _samples.size(); i++)
    {
        std::vector<float> output = run(_samples[i]);

        if (isnan(_threshold))
        {
            uint8_t result =
                std::max_element(output.begin(), output.end()) - output.begin();

            uint8_t label =
                std::max_element(_labels[i].begin(), _labels[i].end()) -
                _labels[i].begin();

            if (result == label)
                success++;
        }
        else
        {
            throw netlib::exception("multiple output test not implemented yet");
        }
    }

    return (float)success / (float)_samples.size();
}