#include <iostream>
#include <vector>

#include "netlib/net.h"
#include "netlib/timer.hpp"

#include "MNIST_test.hpp"

int main(void)
{
    // netlib::timer t1;
    // netlib::net testNet({4, 8, 4}, 0.4, 0.2);
    // testNet.set_random();

    // std::vector<float> floats({0.1, 0.9, 0.1, 0.9});

    // std::vector<uint8_t> ints({0, 1, 0, 1});

    // testNet.print();

    // auto output = testNet.run(floats);

    // for (auto& i : output)
    //     std::cout << (unsigned)i << ' ';

    // std::cout << "\ntraining...\n";

    // {
    //     netlib::timer t2;

    //     for (int i = 0; i < 100000; i++)
    //         testNet.train(floats, ints);
    // }

    // testNet.print();

    // output = testNet.run(floats);

    // for (auto i : output)
    //     std::cout << (unsigned)i << ' ';

    MNIST_test();
}
