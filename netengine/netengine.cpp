#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"

#include "NetEngine/Net.h"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(netengine, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: python_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    py::class_<NetEngine::Net>(m, "Net")
        .def(py::init<const std::vector<size_t>&, float, float>())
        .def("get_eta", &NetEngine::Net::get_eta)
        .def("set_eta", &NetEngine::Net::set_eta)
        .def("get_eta_bias", &NetEngine::Net::get_eta_bias)
        .def("set_eta_bias", &NetEngine::Net::set_eta_bias)
        // .def("__str__", &NetEngine::Net::print)
        .def("n_parameters", &NetEngine::Net::n_parameters)
        .def("set_random", &NetEngine::Net::set_random)
        .def("run", &NetEngine::Net::run)
        .def("train", py::overload_cast<const std::vector<float>&, const std::vector<uint8_t>&>(
                          &NetEngine::Net::train))
        .def("train",
             py::overload_cast<const std::vector<std::vector<float>>&,
                               const std::vector<std::vector<uint8_t>>&, size_t, size_t, size_t,
                               size_t>(&NetEngine::Net::train),
             "samples"_a, "labels"_a, "n_batches"_a, "batch_size"_a, "start_pos"_a = 0,
             "n_threads"_a = std::thread::hardware_concurrency())
        .def("test", &NetEngine::Net::test, "samples"_a, "labels"_a, "subset"_a = 100.0f,
             "threshold"_a = NAN, "n_threads"_a = std::thread::hardware_concurrency());
}