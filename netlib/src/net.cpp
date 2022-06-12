#include "netlib/net.h"
#include "netlib/exceptions.hpp"

#include <iomanip>
#include <iostream>

netlib::net::net(const std::vector<int>& _layout) : m_layout(_layout)
{
    for (int i = 0; i < _layout.size() - 1; i++)
    {
        // add column for bias
        m_weights.push_back(
            new Eigen::MatrixXf(_layout[i + 1], _layout[i] + 1));
    }
}

netlib::net::~net()
{
    for (auto i : m_weights)
        delete i;
}

void netlib::net::print()
{
    std::cout << "##############################################\nlayout: |";

    for (auto i : m_layout)
        std::cout << i << "|";

    std::cout << "\n\nweights:";

    for (auto i : m_weights)
        std::cout << "\n----------------------------------------------\n"
                  << std::fixed << std::setprecision(3) << *i;

    std::cout << "\n----------------------------------------------\n";
}

void netlib::net::set_random()
{
    std::srand((unsigned int)std::time(0));
    for (const auto& i : m_weights)
        i->setRandom();
}

std::vector<float> netlib::net::run(const std::vector<float>& _input)
{
    // check dimensions
    if (_input.size() != m_layout.front())
        throw netlib::dimension_error(_input.size(), m_layout.front());

    // get Eigen vector for input, this does not copy the data
    Eigen::Map<const Eigen::VectorXf> input(_input.data(), _input.size());

    // run first layer
    // note: last column is bias
    auto w = m_weights[0];
    auto r = w->rows();
    auto c = w->cols();

    Eigen::VectorXf temp =
        (w->block(0, 0, r, c - 1) * input + w->block(0, c - 1, r, 1))
            .unaryExpr([](float x) { return (x > 0.0f) ? x : 0.0f; });

    // run remaining layers
    for (int i = 1; i < m_weights.size(); i++)
    {
        w = m_weights[i];
        r = w->rows();
        c = w->cols();

        temp = (w->block(0, 0, r, c - 1) * temp + w->block(0, c - 1, r, 1))
                   .unaryExpr([](float x) { return (x > 0.0f) ? x : 0.0f; });
    }

    return std::vector<float>(temp.data(), temp.data() + temp.size());
}

void netlib::net::train(const std::vector<float>& _input,
                        const std::vector<uint8_t>& _label)
{
    if (_input.size() != m_layout.front())
        throw netlib::dimension_error(_input.size(), m_layout.front());

    if (_label.size() != m_layout.back())
        throw netlib::dimension_error(_label.size(), m_layout.back());

    // Eigen::Map<Eigen::VectorXf> input(_input.data(), _input.size());
    // Eigen::Map<Eigen::Vector<uint8_t, -1>> label(_label.data(),
    // _label.size());

    // Eigen::VectorXf output = run_eigen((Eigen::VectorXf*)&input);

    // Eigen::VectorXf loss = label.cast<float>() - output;

    // backpropagate(&loss);
}
