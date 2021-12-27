#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "my_lib.hpp"
#include "type_definitions.hpp"
#include "wavepacket.hpp"
// #include "integrator.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;

typedef WP::WavePacket WavePacket;

PYBIND11_MODULE(_wavepacket, m) {
  m.doc() = R"pbdoc(
  A python package modelling the interaction of charged particles with an electrostatic pulse.
  -----------------------

  .. currentmodule:: _wavepacket

  .. autosummary::
  :toctree: _generate

  WavePacket
  )pbdoc";


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
    .def("dz", &WavePacket::dz<double>, R"pbdoc(
      Calculate the z derivative
      )pbdoc", py::arg("z"), py::arg("t"))
    .def("dz", &WavePacket::dz<WP::Vector>, R"pbdoc(
        Calculate the z derivative
      )pbdoc", py::arg("z"), py::arg("t"))
    .def("system", &WavePacket::system, R"pbdoc(
        Calculate the single particle dynamic system

        Parameters:
        -----------
        s: array-like, shape(2,)
        The phase space coordinates [z, p]

        Returns:
        --------
        out: array-like, shape(2,)
        the time derivative dsdt
      )pbdoc",py::arg("s"), py::arg("t"))
    .def("_exponent", &WavePacket::_exponent<double>)
    .def("_exponent", &WavePacket::_exponent<WP::Vector>)
    .def("_phase", &WavePacket::_phase<double>)
    .def("_phase", &WavePacket::_phase<WP::Vector>)
    .def("__repr__", &WP::WavePacket::_to_string)
    .def("make_integrator", &WP::WavePacket::make_integrator);
 py::class_<WP::Integrator>(m, "Integrator")
   .def(py::init<WavePacket>())
   .def("integrate", &WP::Integrator::integrate);
}
