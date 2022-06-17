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
        float m_eta_bias;

    public:
        net(const std::vector<int>& _layout, float _eta, float _eta_bias);
        // default _eta_bias to 0.2 * _eta
        net(const std::vector<int>& _layout, float _eta);
        ~net() = default;

        float get_eta();
        void set_eta(float _eta);
        float get_eta_bias();
        void set_eta_bias(float _eta_bias);

        void print();
        void set_random();
        std::vector<float> run(const std::vector<float>& _input);
        void train(const std::vector<float>& _input,
                   const std::vector<uint8_t>& _label);
        // single ouput
        float test(const std::vector<std::vector<float>>& _samples,
                   const std::vector<uint8_t>& _labels);
        // multiple ouputs
        float test(const std::vector<std::vector<float>>& _samples,
                   const std::vector<std::vector<uint8_t>>& _labels);
    };
}

#endif
