#include "NetEngine/Net.h"
#include "NetEngine/Exceptions.h"

#include <cassert>
#include <iomanip>
#include <iostream>

//------------------------------------------------------------------------------
// constructor
//------------------------------------------------------------------------------
NetEngine::Net::Net(const std::vector<size_t> &layout, float eta,
                    float eta_bias)
    : m_layout(layout), m_eta(eta), m_eta_bias(eta_bias) {
    // check min number of layers
    if (layout.size() < 3)
        throw NetEngine::NotEnoughLayers(layout.size());

    // initialize weight matrices, add column for bias
    for (size_t i = 0; i < layout.size() - 1; i++)
        m_weights.push_back(Eigen::MatrixXf(layout[i + 1], layout[i] + 1));
}

NetEngine::Net::Net(const std::vector<size_t> &layout, float eta)
    : Net(layout, eta, 0.2 * eta) {
}

//------------------------------------------------------------------------------
// get, set
//------------------------------------------------------------------------------
float NetEngine::Net::get_eta() {
    return m_eta;
}
void NetEngine::Net::set_eta(float eta) {
    m_eta = eta;
}
float NetEngine::Net::get_eta_bias() {
    return m_eta_bias;
}
void NetEngine::Net::set_eta_bias(float eta_bias) {
    m_eta_bias = eta_bias;
}

//------------------------------------------------------------------------------
// print net
//------------------------------------------------------------------------------
void NetEngine::Net::print() {
    // save stream format
    std::ios streamfmt(nullptr);
    streamfmt.copyfmt(std::cout);

    // print layout
    std::cout << "##############################################\nlayout: |";

    for (auto &i : m_layout)
        std::cout << i << "|";

    // print eta and threshold
    std::cout << "\neta: " << m_eta << "\neta_bias: " << m_eta_bias
              << "\nn_parameters: " << n_parameters();

    // print weights
    std::cout << " \nweights : ";

    for (auto &i : m_weights)
        std::cout << "\n----------------------------------------------\n"
                  << std::fixed << std::setprecision(3) << i;

    std::cout << "\n----------------------------------------------\n";

    // restore stream format
    std::cout.copyfmt(streamfmt);
}

//------------------------------------------------------------------------------
// get number of parameters
//------------------------------------------------------------------------------
size_t NetEngine::Net::n_parameters() {
    size_t n = 0;
    for (const auto &i : m_weights)
        n += i.rows() * i.cols();

    return n;
}

//------------------------------------------------------------------------------
// set weights to random values
//------------------------------------------------------------------------------
void NetEngine::Net::set_random() {
    // seed random number generator to get new random numbers each time
    std::srand((unsigned)std::time(0));

    for (auto &i : m_weights) {
        // set weight matrix to random values in range [-1, 1]
        i.setRandom();

        // change range to [+- 1/sqrt(number of neurons in layer)]
        i / sqrt(i.cols() - 1);
    }
}

//------------------------------------------------------------------------------
// run example
//------------------------------------------------------------------------------
std::vector<float> NetEngine::Net::run(const std::vector<float> &sample) {
    // check dimensions
    if (sample.size() != m_layout.front())
        throw NetEngine::DimensionError(sample.size(), m_layout.front());

    // get Eigen vector for input, this does not copy the data
    Eigen::Map<const Eigen::VectorXf> input(sample.data(), sample.size());

    // run first layer
    // note: last column is bias
    Eigen::VectorXf a =
        (m_weights[0].leftCols(m_weights[0].cols() - 1) * input +
         m_weights[0].col(m_weights[0].cols() - 1))
            .unaryExpr(
                [](float x) { return 0.5f + 0.5f * x / (1.0f + fabsf(x)); });

    // run remaining layers
    for (size_t i = 1; i < m_weights.size(); i++) {
        // note: eval() forces evaluation before new values are assigned to a
        a = ((m_weights[i].leftCols(m_weights[i].cols() - 1) * a).eval() +
             m_weights[i].col(m_weights[i].cols() - 1))
                .unaryExpr([](float x) {
                    return 0.5f + 0.5f * x / (1.0f + fabsf(x));
                });
    }

    return std::vector<float>(a.data(), a.data() + a.size());
}
