#include "netlib/exceptions.hpp"
#include "netlib/net.h"

#include <iostream>
#include <math.h>
#include <thread>

//------------------------------------------------------------------------------
// test accuracy thread
//------------------------------------------------------------------------------
void netlib::net::test_thread(const std::vector<std::vector<float>>& _samples,
                              const std::vector<std::vector<uint8_t>>& _labels,
                              float& _output, size_t _start_pos,
                              size_t _batch_size, float _subset,
                              float _threshold)
{
    // check for bad inputs
    if (_samples.size() != _labels.size())
        throw netlib::set_size_error(_samples.size(), _labels.size());

    if (_start_pos + _batch_size > _samples.size())
        throw netlib::invalid_start_pos(_start_pos);

    if (isnan(_threshold))
    {
        // count correct classifications
        unsigned success = 0;

        if (_subset == 100.0f)
        {
            for (size_t i = _start_pos; i < _start_pos + _batch_size; i++)
            {
                std::vector<float> output = run(_samples[i]);

                uint8_t result =
                    std::max_element(output.begin(), output.end()) -
                    output.begin();

                uint8_t label =
                    std::max_element(_labels[i].begin(), _labels[i].end()) -
                    _labels[i].begin();

                if (result == label)
                    success++;
            }
            _output = (float)success / (float)_batch_size;
        }
        else
        {
            size_t n = _batch_size * (_subset / 100.0f);
            for (size_t i = 0; i < n; i++)
            {
                // seed random number generator
                std::srand(i * (unsigned)std::time(0));

                // choose random sample
                int j = _start_pos + (rand() % (int)_batch_size);

                std::vector<float> output = run(_samples[j]);

                uint8_t result =
                    std::max_element(output.begin(), output.end()) -
                    output.begin();

                uint8_t label =
                    std::max_element(_labels[j].begin(), _labels[j].end()) -
                    _labels[j].begin();

                if (result == label)
                    success++;
            }
            _output = (float)success / (float)n;
        }
    }
    else
    {
        throw netlib::exception("multiple output test not implemented yet");
    }
}

//------------------------------------------------------------------------------
// test accuracy
//------------------------------------------------------------------------------
float netlib::net::test(const std::vector<std::vector<float>>& _samples,
                        const std::vector<std::vector<uint8_t>>& _labels,
                        float _subset, float _threshold, size_t _n_threads)
{
    // std::thread::hardware_concurrency() can return 0
    if (_n_threads == 0)
        _n_threads = 4;

    // check for bad inputs
    if (_samples.size() != _labels.size())
        throw netlib::set_size_error(_samples.size(), _labels.size());

    if (_samples.size() < _n_threads)
        throw netlib::batches_too_small(_samples.size(), _n_threads);

    // storage for threads and results
    std::vector<std::thread> threads;
    threads.reserve(_n_threads);
    std::vector<float> outputs(_n_threads);
    float output = 0;

    // position in _samples and _labels
    size_t pos = 0;

    // number of samples in thread
    std::vector<size_t> n(_n_threads);

    // split samples on threads
    size_t samples_per_thread = _samples.size() / _n_threads;
    size_t remaining_samples = _samples.size() % _n_threads;

    for (size_t i = 0; i < _n_threads; i++)
    {
        if (i < remaining_samples)
            n[i] = samples_per_thread + 1;
        else
            n[i] = samples_per_thread;

        // dispatch thread
        threads.push_back(std::thread(
            &net::test_thread, this, std::ref(_samples), std ::ref(_labels),
            std::ref(outputs[i]), pos, n[i], _subset, _threshold));

        pos += n[i];
    }

    // join threads and evaluate
    for (size_t i = 0; i < _n_threads; i++)
    {
        threads[i].join();

        output += outputs[i] * (float)n[i] / (float)_samples.size();
    }

    return output;
}