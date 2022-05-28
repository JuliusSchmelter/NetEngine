#ifndef NETLIB_EXCEPTIONS_HPP
#define NETLIB_EXCEPTIONS_HPP

#include <exception>
#include <string>

namespace netlib
{
    // base netlib exception
    class exception : public virtual std::exception
    {
    protected:
        std::string m_message;

    public:
        exception()
        {
        }

        exception(const char* message)
        {
            m_message = std::string(message);
        }

        const char* what() const noexcept
        {
            return m_message.c_str();
        }
    };

    // input dimensions do not fit net dimensions
    class dimension_error : public netlib::exception
    {
    public:
        dimension_error(int input_size, int input_layer_size) : exception()
        {
            m_message = "Dimension Error: input vector size (" 
            + std::to_string(input_size) 
            + ") does not match input layer size (" 
            + std::to_string(input_layer_size) 
            + ").";
        }
    };
}

#endif