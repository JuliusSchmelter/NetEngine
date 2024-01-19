#include <cassert>
#include <iomanip>
#include <iostream>

#include "Exceptions.h"
#include "Net.h"

//--------------------------------------------------------------------------------------------------
// Constructor
//--------------------------------------------------------------------------------------------------
NetEngine::Net::Net(const std::vector<size_t>& layout, float eta, float eta_bias)
    : m_layout(layout), m_eta(eta), m_eta_bias(eta_bias) {
    // Check minimum number of layers.
    if (layout.size() < 3)
        throw NetEngine::NotEnoughLayers(layout.size());

    // Initialize weight matrices, add one column for bias.
    for (size_t i = 0; i < layout.size() - 1; i++) {
        size_t rows = layout[i + 1];
        size_t cols = layout[i] + 1;
        float* weights;

        TRY_CUDA(cudaMallocManaged(&weights, rows * cols * sizeof(float)));

        m_weights.push_back({rows, cols, weights});
    }

    set_random();
}

//--------------------------------------------------------------------------------------------------
// Destructor
//--------------------------------------------------------------------------------------------------
NetEngine::Net::~Net() {
    for (auto& i : m_weights)
        TRY_CUDA(cudaFree(i.data));
}

//--------------------------------------------------------------------------------------------------
// get, set
//--------------------------------------------------------------------------------------------------
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

//--------------------------------------------------------------------------------------------------
// Print info string.
//--------------------------------------------------------------------------------------------------
std::string NetEngine::Net::info_string() {
    std::stringstream output;

    output << "layout: ";
    for (size_t i = 0; i < m_layout.size(); i++) {
        output << m_layout[i];
        if (i < m_layout.size() - 1)
            output << " | ";
    }

    output << "\nparameters: " << n_parameters() << "\neta: " << m_eta
           << "\neta_bias: " << m_eta_bias;

    return output.str();
}

//--------------------------------------------------------------------------------------------------
// Get number of parameters.
//--------------------------------------------------------------------------------------------------
size_t NetEngine::Net::n_parameters() {
    size_t n = 0;
    for (const auto& i : m_weights)
        n += i.rows * i.cols;

    return n;
}

//--------------------------------------------------------------------------------------------------
// Set weights to random values.
//--------------------------------------------------------------------------------------------------
void NetEngine::Net::set_random() {
    // seed random number generator to get new random numbers each time
    std::srand((unsigned)std::time(0));

    // set weight matrix to random values in range [+- 1/sqrt(number of neurons in layer)]
    for (auto& i : m_weights)
        for r in i.rows
            for c in i.cols
                i[r * i.cols + c] = 
                    (((float)std::rand() / RAND_MAX)* 2.0f - 1.0f) / sqrt(i.cols - 1);
}