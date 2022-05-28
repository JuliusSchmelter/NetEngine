#ifndef NETLIB_NET_H
#define NETLIB_NET_H

#include "Eigen/Dense"
#include <vector>

namespace netlib
{
    class net
    {
    private:
        std::vector<int> m_layout;
        std::vector<Eigen::MatrixXf*> m_weights;

    public:
        net(std::vector<int> layout);
        ~net();

        void print();
        void set_random();
        std::vector<float> run(std::vector<float> input);
    };
}

#endif