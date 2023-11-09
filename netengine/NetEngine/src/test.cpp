#include "Exceptions.h"
#include "Net.h"

#include <iostream>
#include <math.h>
#include <thread>

//--------------------------------------------------------------------------------------------------
// test accuracy thread
//--------------------------------------------------------------------------------------------------
void NetEngine::Net::test_thread(const std::vector<std::vector<float>>& samples,
                                 const std::vector<std::vector<uint8_t>>& labels, float& output,
                                 size_t start_pos, size_t batch_size, float subset,
                                 float threshold) {
    // check for bad inputs
    if (samples.size() != labels.size())
        throw NetEngine::SetSizeError(samples.size(), labels.size());

    if (start_pos + batch_size > samples.size())
        throw NetEngine::InvalidStartPos(start_pos);

    if (isnan(threshold)) {
        // count correct classifications
        unsigned success = 0;

        if (subset == 100.0f) {
            for (size_t i = start_pos; i < start_pos + batch_size; i++) {
                std::vector<float> output = run(samples[i]);

                uint8_t result = std::max_element(output.begin(), output.end()) - output.begin();

                uint8_t label =
                    std::max_element(labels[i].begin(), labels[i].end()) - labels[i].begin();

                if (result == label)
                    success++;
            }
            output = (float)success / (float)batch_size;
        } else {
            size_t n = batch_size * (subset / 100.0f);
            for (size_t i = 0; i < n; i++) {
                // seed random number generator
                std::srand(i * (unsigned)std::time(0));

                // choose random sample
                int j = start_pos + (rand() % (int)batch_size);

                std::vector<float> output = run(samples[j]);

                uint8_t result = std::max_element(output.begin(), output.end()) - output.begin();

                uint8_t label =
                    std::max_element(labels[j].begin(), labels[j].end()) - labels[j].begin();

                if (result == label)
                    success++;
            }
            output = (float)success / (float)n;
        }
    } else {
        throw NetEngine::Exception("multiple output test not implemented yet");
    }
}

//--------------------------------------------------------------------------------------------------
// test accuracy
//--------------------------------------------------------------------------------------------------
float NetEngine::Net::test(const std::vector<std::vector<float>>& samples,
                           const std::vector<std::vector<uint8_t>>& labels, float subset,
                           float threshold, size_t n_threads) {
    // std::thread::hardware_concurrency() can return 0
    if (n_threads == 0)
        n_threads = 4;

    // check for bad inputs
    if (samples.size() != labels.size())
        throw NetEngine::SetSizeError(samples.size(), labels.size());

    if (samples.size() < n_threads)
        throw NetEngine::BatchesTooSmall(samples.size(), n_threads);

    // storage for threads and results
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    std::vector<float> outputs(n_threads);
    float output = 0;

    // position in samples and labels
    size_t pos = 0;

    // number of samples in thread
    std::vector<size_t> n(n_threads);

    // split samples on threads
    size_t samples_per_thread = samples.size() / n_threads;
    size_t remaining_samples = samples.size() % n_threads;

    for (size_t i = 0; i < n_threads; i++) {
        if (i < remaining_samples)
            n[i] = samples_per_thread + 1;
        else
            n[i] = samples_per_thread;

        // dispatch thread
        threads.push_back(std::thread(&Net::test_thread, this, std::ref(samples), std ::ref(labels),
                                      std::ref(outputs[i]), pos, n[i], subset, threshold));

        pos += n[i];
    }

    // join threads and evaluate
    for (size_t i = 0; i < n_threads; i++) {
        threads[i].join();

        output += outputs[i] * (float)n[i] / (float)samples.size();
    }

    return output;
}