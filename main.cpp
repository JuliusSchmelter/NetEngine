#include <iostream>
#include <vector>

#include "netlib/net.h"

int main(void)
{
    netlib::net testNet({3,2,3,2});
    testNet.set_random();
    testNet.print();

    std::vector<float> input({0.1, 0.2, 0.3});
    std::vector<float> output = testNet.run(input);

    for (auto const& i : output)
        std::cout << i << ' ';
}
