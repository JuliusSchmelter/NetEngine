#include <iostream>
#include <math.h>
#include <thread>

#include "Exceptions.h"
#include "Net.h"

//--------------------------------------------------------------------------------------------------
// Test the net.
float NetEngine::Net::test_eigen(const std::vector<std::vector<float>>& samples,
                                 const std::vector<std::vector<uint8_t>>& labels) {
    // std::thread::hardware_concurrency() can return 0
    size_t n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) {
        n_threads = 4;
        std::cerr
            << "Warning: std::thread::hardware_concurrency() returned 0, using 4 threads instead"
            << std::endl;
    } else if (n_threads > 32) {
        n_threads = 32;
    }

    // Check for bad inputs.
    if (samples.size() < n_threads)
        throw NetEngine::BatchesTooSmall(samples.size(), n_threads);

    if (samples.size() != labels.size())
        throw NetEngine::SetSizeError(samples.size(), labels.size());

    // Storage for threads and results.
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    std::vector<float> outputs(n_threads);
    float output = 0;

    // Position in samples and labels.
    size_t pos = 0;

    // Number of samples in thread.
    std::vector<size_t> n(n_threads);

    // Split samples on threads.
    size_t samples_per_thread = samples.size() / n_threads;
    size_t remaining_samples = samples.size() % n_threads;

    for (size_t i = 0; i < n_threads; i++) {
        if (i < remaining_samples)
            n[i] = samples_per_thread + 1;
        else
            n[i] = samples_per_thread;

        // Dispatch thread.
        threads.push_back(std::thread(&NetEngine::Net::test_worker_eigen, this,
                                      std::ref(outputs[i]), std::ref(samples), std ::ref(labels),
                                      pos, n[i]));

        pos += n[i];
    }

    // Join threads and evaluate.
    for (size_t i = 0; i < n_threads; i++) {
        threads[i].join();
        output += outputs[i] * (float)n[i] / (float)samples.size();
    }

    return output;
}

void NetEngine::Net::test_worker_eigen(float& output,
                                       const std::vector<std::vector<float>>& samples,
                                       const std::vector<std::vector<uint8_t>>& labels,
                                       size_t start_pos, size_t batch_size) {
    // Check for bad inputs.
    if (samples.size() != labels.size())
        throw NetEngine::SetSizeError(samples.size(), labels.size());

    if (start_pos + batch_size > samples.size())
        throw NetEngine::InvalidStartPos(start_pos);

    // Counter for correct classifications.
    unsigned success = 0;

    for (size_t i = start_pos; i < start_pos + batch_size; i++) {
        std::vector<float> output = run_eigen(samples[i]);

        size_t result = std::max_element(output.begin(), output.end()) - output.begin();
        size_t label = std::max_element(labels[i].begin(), labels[i].end()) - labels[i].begin();

        if (result == label)
            success++;
    }
    output = (float)success / (float)batch_size;
}