#include <iostream>
#include <vector>

#include "netlib/net.h"

int main(void)
{
    std::vector<int> layout = {1,2,3,4};
    netlib::net net(layout);
}
