#ifndef NETLIB_NET_H
#define NETLIB_NET_H

#include "Eigen/Dense"
#include <vector>

namespace netlib
{
    class net
    {
    private:
        const std::vector<unsigned> m_layout;
        std::vector<Eigen::MatrixXf> m_weights;
        float m_eta;
        float m_eta_bias;

        void get_weight_mods(
            const std::vector<Eigen::Map<const Eigen::VectorXf>>& _samples,
            const std::vector<Eigen::Map<const Eigen::VectorX<uint8_t>>>&
                _labels,
            std::vector<Eigen::MatrixXf>& _weight_mods);

    public:
        net(const std::vector<unsigned>& _layout, float _eta, float _eta_bias);
        // default _eta_bias to 0.2 * _eta
        net(const std::vector<unsigned>& _layout, float _eta);
        ~net() = default;

        float get_eta();
        void set_eta(float _eta);
        float get_eta_bias();
        void set_eta_bias(float _eta_bias);

        void print();
        size_t n_parameters();
        void set_random();
        std::vector<float> run(const std::vector<float>& _sample);
        // basic training, no mini batching or multithreading
        void train(const std::vector<float>& _sample,
                   const std::vector<uint8_t>& _label);
        // training with mini batching and multithreading
        void train(const std::vector<std::vector<float>>& _samples,
                   const std::vector<std::vector<uint8_t>>& _labels,
                   size_t _batch_size, size_t _n_threads);
        // test accuracy
        float test(const std::vector<std::vector<float>>& _samples,
                   const std::vector<std::vector<uint8_t>>& _labels,
                   float _threshold = NAN);
    };
}

#endif
