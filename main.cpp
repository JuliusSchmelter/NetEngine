#include <iostream>
#include <vector>

#include "netlib/net.h"

int main(void)
{
    netlib::net testNet({3,2,3,2});
    testNet.print();
    testNet.set_random();
    testNet.print();
    std::cout << testNet.run({1.0, 2.0, 3.0});
}
