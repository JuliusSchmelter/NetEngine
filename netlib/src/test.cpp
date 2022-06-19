#include "netlib/exceptions.hpp"
#include "netlib/net.h"

#include <thread>

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

    for (size_t i = 0; i < _samples.size(); i++)
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

//------------------------------------------------------------------------------
// test accuracy thread
//------------------------------------------------------------------------------
void test_thread(const std::vector<std::vector<float>>& _samples,
                 const std::vector<std::vector<uint8_t>>& _labels,
                 float& _ouput, size_t _start_pos, size_t _batch_size,
                 float _subset, float _threshold)
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
            for (size_t i = 0; i < _samples.size(); i++)
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
            _output = (float)success / (float)_samples.size();
        }
        else
        {
        }
    }
    else
    {
        throw netlib::exception("multiple output test not implemented yet");
    }
}

//------------------------------------------------------------------------------
// test accuracy faster
//------------------------------------------------------------------------------
float netlib::net::test(const std::vector<std::vector<float>>& _samples,
                        const std::vector<std::vector<uint8_t>>& _labels,
                        size_t _n_threads, float _subset, float _threshold)
{
    if (_samples.size() != _labels.size())
        throw netlib::set_size_error(_samples.size(), _labels.size());

    unsigned success = 0;

    if (isnan(_threshold))
    {
        for (int i = 0; i < _samples.size(); i++)
        {
            std::vector<float> output = run(_samples[i]);
            uint8_t result =
                std::max_element(output.begin(), output.end()) - output.begin();

            uint8_t label =
                std::max_element(_labels[i].begin(), _labels[i].end()) -
                _labels[i].begin();

            if (result == label)
                success++;
        }
    }
    else
    {
        throw netlib::exception("multiple output test not implemented yet");
    }

    return (float)success / (float)_samples.size();
}