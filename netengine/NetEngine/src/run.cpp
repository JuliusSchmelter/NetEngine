#include "Exceptions.h"
#include "Net.h"

//--------------------------------------------------------------------------------------------------
// Run one sample through the network using Eigen.
//--------------------------------------------------------------------------------------------------
std::vector<float> NetEngine::Net::run_eigen(const std::vector<float>& sample) {
    // Check dimensions.
    if (sample.size() != m_layout.front())
        throw NetEngine::DimensionError(sample.size(), m_layout.front());

    // Get Eigen vector for input, this does not copy the data.
    Eigen::Map<const Eigen::VectorXf> input(sample.data(), sample.size());

    // Run first layer.
    // Note: last column is bias.
    Eigen::VectorXf a = (m_weights_eigen[0].leftCols(m_weights_eigen[0].cols() - 1) * input +
                         m_weights_eigen[0].col(m_weights_eigen[0].cols() - 1))
                            .unaryExpr([](float x) { return 0.5f + 0.5f * x / (1.0f + fabsf(x)); });

    // Run remaining layers.
    for (size_t i = 1; i < m_weights_eigen.size(); i++) {
        // note: eval() forces evaluation before new values are assigned to a
        a = ((m_weights_eigen[i].leftCols(m_weights_eigen[i].cols() - 1) * a).eval() +
             m_weights_eigen[i].col(m_weights_eigen[i].cols() - 1))
                .unaryExpr([](float x) { return 0.5f + 0.5f * x / (1.0f + fabsf(x)); });
    }

    return std::vector<float>(a.data(), a.data() + a.size());
}