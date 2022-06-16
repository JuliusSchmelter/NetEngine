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
        exception() : m_message("Generic Netlib Error: no message.")
        {
        }

        exception(std::string _message) : m_message(_message)
        {
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
        dimension_error(size_t _input_size, size_t _input_layer_size)
            : exception()
        {
            m_message = "Dimension Error: input vector size (" +
                        std::to_string(_input_size) +
                        ") does not match input layer size (" +
                        std::to_string(_input_layer_size) + ").";
        }
    };

    // comparing sets with different sizes
    class set_size_error : public netlib::exception
    {
    public:
        set_size_error(size_t _size1, size_t _size2) : exception()
        {
            m_message = "Set Size Error: comparing set of size " +
                        std::to_string(_size1) + " to set of size " +
                        std::to_string(_size2) + ".";
        }
    };
}

#endif