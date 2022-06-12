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
        std::vector<Eigen::MatrixXf> m_weights;
        float m_eta;
        float m_threshold;

    public:
        net(const std::vector<int>& _layout, float _eta, float _threshold);
        ~net() = default;

        float get_eta();
        void set_eta(float _threshold);
        float get_threshold();
        void set_threshold(float _threshold);

        void print();
        void set_random();
        std::vector<uint8_t> run(const std::vector<float>& _input);
        void train(const std::vector<float>& _input,
                   const std::vector<uint8_t>& _label);
    };
}

#endif