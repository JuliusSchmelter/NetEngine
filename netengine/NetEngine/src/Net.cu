#include <cassert>
#include <iomanip>
#include <iostream>

#include "Exceptions.h"
#include "Net.h"

// CUDA kernel to set one float value in device memory.
__global__ void set_value(float* data, float value) {
    *data = value;
}

//--------------------------------------------------------------------------------------------------
// Constructor
//--------------------------------------------------------------------------------------------------
NetEngine::Net::Net(const std::vector<uint32_t>& layout, float eta, float eta_bias)
    : m_layout(layout), m_eta(eta), m_eta_bias(eta_bias) {
    // Check minimum number of layers.
    if (layout.size() < 3)
        throw NetEngine::NotEnoughLayers(layout.size());

    // Seed random number generator to get new random numbers each time.
    std::srand((unsigned)std::time(0));

    // Initialize weight matrices, add one column for bias.
    for (size_t i = 0; i < layout.size() - 1; i++) {
        float* weights;
        uint32_t rows = layout[i + 1];
        uint32_t cols = layout[i] + 1;

        TRY_CUDA(cudaMalloc(&weights, rows * cols * sizeof(float)));

        // Set weight matrix to random values in range [+- 1/sqrt(number of neurons in layer)].
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t c = 0; c < cols; c++)
                set_value<<<1, 1>>>(&weights[r * cols + c],
                                    (((float)std::rand() / RAND_MAX) * 2.0f - 1.0f) /
                                        sqrt(cols - 1));

        TRY_CUDA(cudaDeviceSynchronize());

        CudaMatrix cuda_matrix = {weights, rows, cols};
        m_weights.push_back(cuda_matrix);
    }
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
    for (auto& i : m_weights)
        n += i.rows * i.cols;

    return n;
}