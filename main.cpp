#include <iostream>
#include <vector>

#include "netlib/net.h"
#include "netlib/timer.hpp"

int main(void)
{
    netlib::timer t;
    netlib::net testNet({20, 5, 20}, 0.1, 0.5);
    testNet.set_random();

    std::vector<float> floats_20({0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1,
                                  0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9,
                                  0.1, 0.9, 0.1, 0.9, 0.1, 0.9});

    std::vector<uint8_t> ints_20(
        {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1});

    testNet.print();

    auto output = testNet.run(floats_20);

    for (auto& i : output)
        std::cout << (unsigned)i << ' ';

    std::cout << std::endl;

    for (int i = 0; i < 1; i++)
        testNet.train(floats_20, ints_20);

    // output = testNet.run(floats_20);

    // for (auto i : output)
    //     std::cout << i << ' ';
}
