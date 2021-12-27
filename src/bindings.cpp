#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "my_lib.hpp"
#include "type_definitions.hpp"
#include "wavepacket.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;

typedef WP::WavePacket WavePacket;

PYBIND11_MODULE(_wavepacket, m) {
  m.doc() = R"pbdoc(
  Pybind11 example plugin
  -----------------------

  .. currentmodule:: scikit_build_example

  .. autosummary::
  :toctree: _generate

  add
  subtract
  )pbdoc";

  m.def("add", static_cast<int(*)(int,int)>(&mylib::add), R"pbdoc(
  Add two numbers

  Some other explanation about the add function.
  )pbdoc", py::arg("i"), py::arg("j")=3);

  m.def("add", static_cast<double(*)(double,double)>(&mylib::add), R"pbdoc(
  Add two numbers

  Some other explanation about the add function.
  )pbdoc", py::arg("i"), py::arg("j")=3.0);


  m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
  Subtract two numbers

  Some other explanation about the subtract function.
  )pbdoc", py::arg("i"), py::arg("j"));

  #ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
  #else
  m.attr("__version__") = "dev";
  #endif

  py::class_<WavePacket>(m, "WavePacket")
    .def_readonly("A",&WavePacket::A)
    .def_readonly("sigma",&WavePacket::sigma)
    .def_readonly("k",&WavePacket::k)
    .def_readonly("vp",&WavePacket::vp)
    .def(py::init<double, double, double, double>(),
          py::arg("A"),
          py::arg("sigma"),
          py::arg("k"),
          py::arg("vp"))
    .def("__call__", &WavePacket::operator()<double>, py::arg("z"), py::arg("t"))
    .def("__call__", &WavePacket::operator()<WP::Vector>, py::arg("z"), py::arg("t"))
    .def("dz", &WavePacket::dz<double>, py::arg("z"), py::arg("t"))
    .def("dz", &WavePacket::dz<WP::Vector>, py::arg("z"), py::arg("t"))
    .def("system", &WavePacket::system, py::arg("s"), py::arg("t"))
    .def("_exponent", &WavePacket::_exponent<double>)
    .def("_exponent", &WavePacket::_exponent<WP::Vector>)
    .def("_phase", &WavePacket::_phase<double>)
    .def("_phase", &WavePacket::_phase<WP::Vector>)
    .def("__repr__", &WP::WavePacket::_to_string);



}
