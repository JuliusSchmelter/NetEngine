#ifndef NETLIB_NET_H
#define NETLIB_NET_H

#include "Eigen/Dense"
#include <vector>

namespace netlib
{
    class net
    {
    private:
        const std::vector<int> m_layout;
        std::vector<Eigen::MatrixXf*> m_weights;

    public:
        net(const std::vector<int>& _layout);
        ~net();

        void print();
        void set_random();
        std::vector<float> run(const std::vector<float>& _input);
        void train(const std::vector<float>& _input,
                   const std::vector<uint8_t>& _label);
    };
}

#endif