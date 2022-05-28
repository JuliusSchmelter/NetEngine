#ifndef NET_H
#define NET_H

#include "Eigen/Dense"
#include <vector>

namespace netlib
{
    class net
    {
    private:
        std::vector<Eigen::MatrixXf*> m_weights;
    public:
        net(std::vector<int> layout);
        ~net();
    };
}

#endif