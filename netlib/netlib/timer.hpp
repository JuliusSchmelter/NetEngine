#ifndef NETLIB_TIMER_HPP
#define NETLIB_TIMER_HPP

#include <chrono>
#include <iostream>

namespace netlib
{
    class timer
    {
    private:
        std::chrono::_V2::system_clock::time_point m_start;
        std::string m_name;

    public:
        timer()
            : m_start(std::chrono::high_resolution_clock::now()),
              m_name("timer")
        {
        }

        timer(std::string _name)
            : m_start(std::chrono::high_resolution_clock::now()), m_name(_name)
        {
        }

        ~timer()
        {
            std::chrono::duration<float> duration =
                std::chrono::high_resolution_clock::now() - m_start;

            std::cout << "[" << m_name << " scope lived "
                      << duration.count() * 1000.0f << " ms]" << std::endl;
        }
    };
}

#endif