#include "netlib/net.h"

#include <iostream>

#include "Eigen/Dense"

void netlib::test()
{
    Eigen::Matrix2d test;
    test << 1, 2, 3, 4;
    std::cout << test;
}
