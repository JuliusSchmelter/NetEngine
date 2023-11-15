#include "netengine/NetEngine/DevTools.h"
#include "netengine/NetEngine/Net.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#define N_TEST 10000
#define N_TRAIN 60000

#define LAYOUT                                                                                     \
    { 28 * 28, 100, 10 }

#define ETA 0.1
#define ETA_BIAS 0.02
#define MINI_BATCHES 8
#define MINI_BATCH_SIZE 256
#define N_THREADS 12

int main() {
    TIMER(main)

    // init net
    NetEngine::Net net(LAYOUT, ETA, ETA_BIAS);
    LOG(net.n_parameters())
    net.set_random();

    // define targets
    std::vector<std::vector<uint32_t>> target = {
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};

    // get storage
    std::vector<std::vector<float>> test_images(N_TEST);
    std::vector<std::vector<uint32_t>> test_labels(N_TEST);
    std::vector<std::vector<float>> train_images(N_TRAIN);
    std::vector<std::vector<uint32_t>> train_labels(N_TRAIN);

    // load test data
    TIC(load_data)
    std::ifstream ifstream;
    ifstream.open("data/mnist/test_images", std::ios::binary);
    assert(ifstream);
    ifstream.seekg(16); // skip header
    for (size_t i = 0; i < N_TEST; i++) {
        test_images[i] = std::vector<float>(28 * 28);
        for (size_t j = 0; j < 28 * 28; j++) {
            uint8_t tmp;
            ifstream.read((char*)&tmp, 1);
            test_images[i][j] = tmp / 255.0f; // scale to [0, 1]
        }
    }
    ifstream.close();

    ifstream.open("data/mnist/test_labels", std::ios::binary);
    assert(ifstream);
    ifstream.seekg(8); // skip header
    for (int i = 0; i < N_TEST; i++) {
        uint8_t tmp;
        ifstream.read((char*)&tmp, 1);
        test_labels[i] = target[tmp];
    }
    ifstream.close();

    // load training data
    ifstream.open("data/mnist/train_images", std::ios::binary);
    assert(ifstream);
    ifstream.seekg(16); // skip header
    for (size_t i = 0; i < N_TRAIN; i++) {
        train_images[i].reserve(28 * 28);
        for (int j = 0; j < 28 * 28; j++) {
            uint8_t tmp;
            ifstream.read((char*)&tmp, 1);
            train_images[i].push_back(tmp / 255.0f); // scale to [0, 1]
        }
    }
    ifstream.close();

    ifstream.open("data/mnist/train_labels", std::ios::binary);
    assert(ifstream);
    ifstream.seekg(8); // skip header
    for (size_t i = 0; i < N_TRAIN; i++) {
        uint8_t tmp;
        ifstream.read((char*)&tmp, 1);
        train_labels[i] = target[tmp];
    }
    ifstream.close();
    TOC(load_data)

    // train and test
    for (size_t trained = 0;; trained += MINI_BATCHES * MINI_BATCH_SIZE) {
        LOG(trained)
        TIC(train);
        net.train(train_images, train_labels, MINI_BATCHES, MINI_BATCH_SIZE, trained % N_TRAIN,
                  N_THREADS);
        TOC(train)

        TIC(test)
        std::cout << "accuracy: " << 100 * net.test(test_images, test_labels) << "%\n";
        TOC(test)
    }
}