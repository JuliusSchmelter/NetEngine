#include "netlib/net.h"


netlib::net::net(std::vector<int> layout)
{
    for (int i = 0; i < layout.size() - 1; i++)
    {
        m_weights.push_back(new Eigen::MatrixXf(layout[i+1], layout[i]));
    }
}

netlib::net::~net()
{
    for (const auto& i: m_weights) delete i;
}


