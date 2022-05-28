#include "netlib/net.h"
#include "netlib/exceptions.hpp"

#include <iostream>

netlib::net::net(std::vector<int> layout)
{
    m_layout = layout;
    for (int i = 0; i < layout.size() - 1; i++)
    {
        m_weights.push_back(new Eigen::MatrixXf(layout[i+1], layout[i]));
    }
}

netlib::net::~net()
{
    for (const auto& i: m_weights) delete i;
}

void netlib::net::print()
{
    std::cout << "|";
    for (const auto& i: m_layout) std::cout << i << "|";
    std::cout << std::endl;
    for (const auto& i: m_weights)
        std::cout << "----------\n" << *i << std::endl;
}

void netlib::net::set_random()
{
    for (const auto& i: m_weights) i->setRandom();
}

std::vector<float> netlib::net::run(std::vector<float> input)
{
    if (input.size() != m_layout[0])
        throw netlib::dimension_error(input.size(), m_layout[0]);

}
