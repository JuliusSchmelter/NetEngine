#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/stl.h"

#include "NetEngine/Net.h"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(netengine, m) {
    py::class_<NetEngine::Net>(m, "Net")
        .def(py::init<const std::vector<uint32_t>&, float, float, bool>(), "layout"_a, "eta"_a,
             "eta_bias"_a, "try_cuda"_a = true)
        .def("get_eta", &NetEngine::Net::get_eta)
        .def("set_eta", &NetEngine::Net::set_eta)
        .def("get_eta_bias", &NetEngine::Net::get_eta_bias)
        .def("set_eta_bias", &NetEngine::Net::set_eta_bias)
        .def("cuda_enabled", &NetEngine::Net::cuda_enabled)
        .def("__str__", &NetEngine::Net::info_string)
        .def("n_parameters", &NetEngine::Net::n_parameters)
        .def("run", &NetEngine::Net::run)
        .def("train", &NetEngine::Net::train, "samples"_a, "labels"_a, "n_samples"_a,
             "start_pos"_a = 0)
        .def("test", &NetEngine::Net::test);
}