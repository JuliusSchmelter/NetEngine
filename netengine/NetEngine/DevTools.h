#ifndef NETENGINE_DEVTOOLS_H
#define NETENGINE_DEVTOOLS_H

#include <chrono>
#include <iomanip>
#include <iostream>

#define LOG(var) std::cout << #var << " = " << var << std::endl;
#define LOGS(string) std::cout << #string << std::endl;
#define TIMER(name) NetEngine::Timer timer_##name(#name);
#define TIC(name) NetEngine::Timer* timer_##name = new NetEngine::Timer(#name);
#define TOC(name) delete timer_##name;

namespace NetEngine {
    class Timer {
    private:
        std::chrono::_V2::system_clock::time_point m_start;
        std::string m_name;

    public:
        Timer() : m_name("timer") {
            m_start = std::chrono::high_resolution_clock::now();
        }

        Timer(const char* name) : m_name(name) {
            m_start = std::chrono::high_resolution_clock::now();
        }

        ~Timer() {
            std::chrono::duration<float> duration =
                std::chrono::high_resolution_clock::now() - m_start;

            // save stream format
            std::ios streamfmt(nullptr);
            streamfmt.copyfmt(std::cout);

            // less than 100 ms
            if (duration < std::chrono::duration<float>(0.1f))
                std::cout << std::fixed << std::setprecision(2) << "[" << m_name << " scope lived "
                          << duration.count() * 1000.0f << " ms]" << std::endl;

            // less than 1 s
            else if (duration < std::chrono::duration<float>(1.0f))
                std::cout << std::fixed << std::setprecision(0) << "[" << m_name << " scope lived "
                          << duration.count() * 1000.0f << " ms]" << std::endl;
            // more than 1 s
            else
                std::cout << std::fixed << std::setprecision(2) << "[" << m_name << " scope lived "
                          << duration.count() << " s]" << std::endl;

            // restore stream format
            std::cout.copyfmt(streamfmt);
        }
    };
}

#endif // NETENGINE_DEVTOOLS_H