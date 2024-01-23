#include <algorithm>
#include <iostream>
#include <math.h>
#include <thread>

#include "Exceptions.h"
#include "Net.h"

//--------------------------------------------------------------------------------------------------
// Test accuracy.
//--------------------------------------------------------------------------------------------------
float NetEngine::Net::test_cuda(const std::vector<std::vector<float>>& samples,
                                const std::vector<std::vector<uint8_t>>& labels) {
    // Check for bad inputs.
    if (samples.size() != labels.size())
        throw NetEngine::SetSizeError(samples.size(), labels.size());

    // Allocate device memory for (intermediate) results.
    float* results_d[m_weights_cuda.size()];
    for (size_t i = 0; i < m_weights_cuda.size(); i++) {
        TRY_CUDA(cudaMalloc(&results_d[i], m_weights_cuda[i].rows * sizeof(float)));
    }

    // Allocate device memory for sample.
    float* sample_d;
    TRY_CUDA(cudaMalloc(&sample_d, samples[0].size() * sizeof(float)));

    // Allocate host memory for output.
    std::vector<float> output(m_layout.back());

    // Iterate samples.
    size_t success = 0;
    for (size_t i = 0; i < samples.size(); i++) {
        // Copy sample to device memory.
        TRY_CUDA(cudaMemcpy(sample_d, samples[i].data(), samples[i].size() * sizeof(float),
                            cudaMemcpyHostToDevice));

        // Run sample on network.
        run_cuda_dev_ptrs(sample_d, results_d);

        // Copy output to host.
        TRY_CUDA(cudaMemcpy(output.data(), results_d[m_weights_cuda.size() - 1],
                            m_layout.back() * sizeof(float), cudaMemcpyDeviceToHost));

        // Check result.
        size_t result = std::max_element(output.begin(), output.end()) - output.begin();
        size_t label = std::max_element(labels[i].begin(), labels[i].end()) - labels[i].begin();

        if (result == label)
            success++;
    }

    // Free memory.
    TRY_CUDA(cudaFree(sample_d));
    for (size_t i = 0; i < m_weights_cuda.size(); i++) {
        TRY_CUDA(cudaFree(results_d[i]));
    }

    return (float)success / (float)samples.size();
}
