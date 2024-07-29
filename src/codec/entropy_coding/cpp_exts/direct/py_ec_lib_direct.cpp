#include <pybind11/pybind11.h>
#include "ec_lib_direct.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(ec_direct, m) {
    m.doc() = "EC Lib for direct encoding bits to a bitstream";

    py::class_<EcLibDirect>(m, "EcLibDirect")
        .def(py::init([] (py::buffer &memory) {
            auto infoMemory = memory.request();
            return std::unique_ptr<EcLibDirect>(new EcLibDirect(
                (uint8_t*)infoMemory.ptr, infoMemory.size
            ));
        }))
        .def("read_bits", [](EcLibDirect& coder, uint8_t bits_count) { uint32_t ans = 0; uint8_t output_size = coder.read_bits(ans, bits_count); return std::make_tuple(output_size, ans); } )
        .def("write_bits", &EcLibDirect::write_bits)
        .def("set_pointer", &EcLibDirect::set_pointer)
        .def("get_pointer_pos", &EcLibDirect::get_pointer_pos)

        .def("clear", &EcLibDirect::clear)
    ;

}
