/* The copyright in this software is being made available under the BSD
 License, included below. This software may be subject to other third party
 and contributor rights, including patent rights, and no such rights are
 granted under this license.

 Copyright (c) 2010-2022, ITU/ISO/IEC
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.
 * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 be used to endorse or promote products derived from this software without
 specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 THE POSSIBILITY OF SUCH DAMAGE. */

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
