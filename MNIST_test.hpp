#include <cassert>
#include <fstream>

void MNIST_test()
{
    std::cout << "MNIST_test\n";

    // define targets
    std::vector<std::vector<uint8_t>> target = {
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};

    // get storage
    std::vector<uint8_t> test_labels(10000);
    std::vector<std::vector<float>> test(10000);
    std::vector<uint8_t> train_labels(60000);
    std::vector<std::vector<float>> train(60000);

    {
        netlib::timer t("load data");

        // load test data
        std::ifstream labels;
        labels.open("C:/Code/BigEarthNet/data/mnist/test_labels",
                    std::ios::binary);
        assert(labels);
        labels.seekg(8); // skip header
        labels.read((char*)test_labels.data(), 10000);
        labels.close();

        std::ifstream examples;
        examples.open("C:/Code/BigEarthNet/data/mnist/test", std::ios::binary);
        assert(examples);
        examples.seekg(16); // skip header
        for (int i = 0; i < 10000; i++)
        {
            test[i] = std::vector<float>(28 * 28);
            for (int j = 0; j < 28 * 28; j++)
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
        labels.read((char*)train_labels.data(), 60000);
        labels.close();

        examples.open("C:/Code/BigEarthNet/data/mnist/train", std::ios::binary);
        assert(examples);
        examples.seekg(16); // skip header
        for (int i = 0; i < 60000; i++)
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

    // init net
    netlib::net net({28 * 28, 20, 10}, 0.001);
    net.set_random();

    {
        netlib::timer t("test accuracy");

        // test accuracy
        std::cout << 100 * net.test(test, test_labels) << '\n';
    }

    {
        netlib::timer t("train net");

        // train net
        for (int i = 0; i < 0; i++)
            net.train(train[i], target[train_labels[i]]);
    }

    // test accuracy
    // std::cout << 100 * net.test(test, test_labels) << '\n';

    // for (auto i : test[1234])
    //     std::cout << i << ' ';

    // net.print();

    auto output = net.run(test[1234]);
    for (auto i : output)
        std::cout << i << ' ';

    uint8_t res =
        std::max_element(output.begin(), output.end()) - output.begin();

    std::cout << '\n' << res;
}