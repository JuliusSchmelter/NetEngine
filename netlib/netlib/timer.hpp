#ifndef NETLIB_TIMER_HPP
#define NETLIB_TIMER_HPP

#include <chrono>
#include <iostream>

namespace netlib
{
    class timer
    {
    private:
        std::chrono::_V2::system_clock::time_point start;

    public:
        timer()
        {
            start = std::chrono::high_resolution_clock::now();
        }

        ~timer()
        {
            std::chrono::duration<float> duration =
                std::chrono::high_resolution_clock::now() - start;
            std::cout << "[timer lived " << duration.count() * 1000.0f << " ms]"
                      << std::endl;
        }
    };
}

#endif