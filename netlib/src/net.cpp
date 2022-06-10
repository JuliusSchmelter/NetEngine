#include "netlib/net.h"
#include "netlib/exceptions.hpp"

#include <iomanip>
#include <iostream>

netlib::net::net(std::vector<int> layout)
{
    m_layout = layout;
    for (int i = 0; i < layout.size() - 1; i++)
    {
        m_weights.push_back(new Eigen::MatrixXf(layout[i + 1], layout[i]));
    }
}

netlib::net::~net()
{
    for (const auto& i : m_weights)
        delete i;
}

void netlib::net::print()
{
    std::cout << "##############################################\nlayout: |";

    for (const auto& i : m_layout)
        std::cout << i << "|";

    std::cout << "\n\nweights:";

    for (const auto& i : m_weights)
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

float netlib::net::ReLU(float _x)
{
    return (_x > 0.0f) ? _x : 0.0f;
}

std::vector<float> netlib::net::run(std::vector<float>& _input)
{
    if (_input.size() != m_layout[0])
        throw netlib::dimension_error(_input.size(), m_layout[0]);

    Eigen::Map<Eigen::VectorXf> input(_input.data(), _input.size());

    Eigen::VectorXf temp =
        (*m_weights[0] * input).unaryExpr([](float x) { return (x > 0.0f) ? x : 0.0f; });

    for (int i = 1; i < m_weights.size(); i++)
        temp = (*m_weights[i] * temp).unaryExpr([](float x) { return (x > 0.0f) ? x : 0.0f; });

    return std::vector<float>(temp.data(), temp.data() + temp.size());
}
