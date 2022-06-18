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

    // vector dimensions do not fit net dimensions
    class dimension_error : public netlib::exception
    {
    public:
        dimension_error(size_t _vector_size, size_t _layer_size)
            : exception()
        {
            m_message = "Dimension Error: vector size (" +
                        std::to_string(_vector_size) +
                        ") does not match layer size (" +
                        std::to_string(_layer_size) + ").";
        }
    };

    // comparing sets of different sizes
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

    // nets with less than three layers are not supported
    class not_enough_layers : public netlib::exception
    {
    public:
        not_enough_layers(size_t _n_layers) : exception()
        {
            m_message =
                "Not Enough Layers: nets with less than three layers are "
                "not supported (tried " +
                std::to_string(_n_layers) + " layers).";
        }
    };
}

#endif