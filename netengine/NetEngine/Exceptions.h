#ifndef NETENGINE_EXCEPTIONS_H
#define NETENGINE_EXCEPTIONS_H

#include <exception>
#include <string>

namespace NetEngine {
    class Exception : public virtual std::exception {
    protected:
        std::string m_message;

    public:
        Exception() : m_message("Generic NetEngine Error: no message.") {
        }

        Exception(std::string message) : m_message(message) {
        }

        const char* what() const noexcept {
            return m_message.c_str();
        }
    };

    class DimensionError : public NetEngine::Exception {
    public:
        DimensionError(size_t vector_size, size_t layer_size) : Exception() {
            m_message = "Dimension Error: vector size (" + std::to_string(vector_size) +
                        ") does not match layer size (" + std::to_string(layer_size) + ").";
        }
    };

    class SetSizeError : public NetEngine::Exception {
    public:
        SetSizeError(size_t size1, size_t size2) : Exception() {
            m_message = "Set Size Error: comparing set of size " + std::to_string(size1) +
                        " to set of size " + std::to_string(size2) + ".";
        }
    };

    class NotEnoughLayers : public NetEngine::Exception {
    public:
        NotEnoughLayers(size_t n_layers) : Exception() {
            m_message = "Layout Error: nets with less than three layers are "
                        "not supported (tried " +
                        std::to_string(n_layers) + " layers).";
        }
    };

    class BatchesTooSmall : public NetEngine::Exception {
    public:
        BatchesTooSmall(size_t batch_size, size_t n_threads) : Exception() {
            m_message = "Batch Size Error: batch size (" + std::to_string(batch_size) +
                        ") is smaller than number of threads (" + std::to_string(n_threads) + ").";
        }
    };

    class BatchesTooLarge : public NetEngine::Exception {
    public:
        BatchesTooLarge(size_t batch_size, size_t n_samples) : Exception() {
            m_message = "Batch Size Error: batch size (" + std::to_string(batch_size) +
                        ") is larger than number of samples (" + std::to_string(n_samples) + ").";
        }
    };

    class InvalidStartPos : public NetEngine::Exception {
    public:
        InvalidStartPos(size_t start_pos) : Exception() {
            m_message = "Range Error: start position (" + std::to_string(start_pos) +
                        ") is not at least one batch size smaller than number "
                        "of samples.";
        }
    };
}

#endif // NETENGINE_EXCEPTIONS_H