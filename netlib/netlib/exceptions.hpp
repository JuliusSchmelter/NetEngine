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

    class dimension_error : public netlib::exception
    {
    public:
        dimension_error(size_t _vector_size, size_t _layer_size) : exception()
        {
            m_message = "Dimension Error: vector size (" +
                        std::to_string(_vector_size) +
                        ") does not match layer size (" +
                        std::to_string(_layer_size) + ").";
        }
    };

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

    class not_enough_layers : public netlib::exception
    {
    public:
        not_enough_layers(size_t _n_layers) : exception()
        {
            m_message = "Layout Error: nets with less than three layers are "
                        "not supported (tried " +
                        std::to_string(_n_layers) + " layers).";
        }
    };

    class batches_too_small : public netlib::exception
    {
    public:
        batches_too_small(size_t _batch_size, size_t _n_threads) : exception()
        {
            m_message = "Batch Size Error: batch size (" +
                        std::to_string(_batch_size) +
                        ") is smaller than number of threads (" +
                        std::to_string(_n_threads) + ").";
        }
    };

    class batches_too_large : public netlib::exception
    {
    public:
        batches_too_large(size_t _batch_size, size_t _n_samples) : exception()
        {
            m_message = "Batch Size Error: batch size (" +
                        std::to_string(_batch_size) +
                        ") is larger than number of samples (" +
                        std::to_string(_n_samples) + ").";
        }
    };
}

#endif