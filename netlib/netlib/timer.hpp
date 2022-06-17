#ifndef NETLIB_TIMER_HPP
#define NETLIB_TIMER_HPP

#include <chrono>
#include <iomanip>
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

            // save stream format
            std::ios streamfmt(nullptr);
            streamfmt.copyfmt(std::cout);

            // less than 100 ms
            if (duration < std::chrono::duration<float>(0.1f))
                std::cout << std::fixed << std::setprecision(2) << "[" << m_name
                          << " scope lived " << duration.count() * 1000.0f
                          << " ms]" << std::endl;
            // less than 1 s
            else if (duration < std::chrono::duration<float>(1.0f))
                std::cout << std::fixed << std::setprecision(0) << "[" << m_name
                          << " scope lived " << duration.count() * 1000.0f
                          << " ms]" << std::endl;
            // more than 1 s
            else
                std::cout << std::fixed << std::setprecision(2) << "[" << m_name
                          << " scope lived " << duration.count() << " s]"
                          << std::endl;

            // restore stream format
            std::cout.copyfmt(streamfmt);
        }
    };
}

#endif