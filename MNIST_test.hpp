#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#include "netlib/net.h"
#include "netlib/timer.hpp"

#define N_TEST 10000
#define N_TRAIN 60000

void MNIST_test()
{
    std::cout << "MNIST_test\n";
    netlib::timer t_master("master");

    //------------------------------------------------------------------------------

    // init net
    netlib::net net({28 * 28, 300, 300, 50, 10}, 0.005);
    net.set_random();
    std::cout << "n_parameters: " << net.n_parameters() << '\n';

    // define targets
    std::vector<std::vector<uint8_t>> target = {
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};

    //------------------------------------------------------------------------------

    // get storage
    std::vector<std::vector<uint8_t>> test_labels(N_TEST);
    std::vector<std::vector<float>> test(N_TEST);
    std::vector<std::vector<uint8_t>> train_labels(N_TRAIN);
    std::vector<std::vector<float>> train(N_TRAIN);

    {
        netlib::timer t("load data");

        // load test data
        std::ifstream labels;
        labels.open("C:/Code/BigEarthNet/data/mnist/test_labels",
                    std::ios::binary);
        assert(labels);
        labels.seekg(8); // skip header
        for (int i = 0; i < N_TEST; i++)
        {
            uint8_t tmp;
            labels.read((char*)&tmp, 1);
            test_labels[i] = target[tmp];
        }
        labels.close();

        std::ifstream examples;
        examples.open("C:/Code/BigEarthNet/data/mnist/test", std::ios::binary);
        assert(examples);
        examples.seekg(16); // skip header
        for (size_t i = 0; i < N_TEST; i++)
        {
            test[i] = std::vector<float>(28 * 28);
            for (size_t j = 0; j < 28 * 28; j++)
            {
                uint8_t tmp;
                examples.read((char*)&tmp, 1);
                test[i][j] = tmp / 255.0f; // scale to [0, 1]
            }
        }
        examples.close();

        // load training data
        labels.open("C:/Code/BigEarthNet/data/mnist/train_labels",
                    std::ios::binary);
        assert(labels);
        labels.seekg(8); // skip header
        for (size_t i = 0; i < N_TRAIN; i++)
        {
            uint8_t tmp;
            labels.read((char*)&tmp, 1);
            train_labels[i] = target[tmp];
        }
        labels.close();

        examples.open("C:/Code/BigEarthNet/data/mnist/train", std::ios::binary);
        assert(examples);
        examples.seekg(16); // skip header
        for (size_t i = 0; i < N_TRAIN; i++)
        {
            train[i].reserve(28 * 28);
            for (int j = 0; j < 28 * 28; j++)
            {
                uint8_t tmp;
                examples.read((char*)&tmp, 1);
                train[i].push_back(tmp / 255.0f); // scale to [0, 1]
            }
        }
        examples.close();
    }

    //--------------------------------------------------------------------------
    // train and test

    int mini_batches = 20;
    int m_b_size = 480;

    for (size_t trained = 0;; trained += mini_batches * m_b_size)
    {
        std::cout << "trained: " << trained << '\n';
        
        {
        netlib::timer t("train");

        net.train(train, train_labels, mini_batches, m_b_size,
                  trained % N_TRAIN, 3);
        }

        {
            netlib::timer t("test");
            
            std::cout << "accuracy: " << 100 * net.test(test, test_labels)
                      << "%\n";
        }
    }
}