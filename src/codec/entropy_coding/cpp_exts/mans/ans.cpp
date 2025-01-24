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
#include "compressor.h"
#include "decompressor.h"
namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(ans, m) {
    m.doc() = "me-tANS encoder for mass bits 8";

    py::class_<ANSEncoder>(m, "ANSEncoder")
        .def(py::init<ssize_t, int>())
        .def("set_sgm_transitions", [] (ANSEncoder &compressor, const py::buffer &transitions, const py::buffer &bounds, const py::buffer &stateMaps) {
            auto info_transition = transitions.request();
            auto info_bound = bounds.request();
            auto info_stateMap = stateMaps.request();
            compressor.set_sgm_transitions((const uint32_t*)info_transition.ptr, (const uint8_t*)info_bound.ptr, 
                (const uint8_t*)info_stateMap.ptr, info_transition.shape[0], info_transition.shape[1]);
        })
        .def("encode_sgm", [] (ANSEncoder &compressor, const py::buffer &sigmaIdx, py::buffer &r, const py::buffer &masks) {
            auto info_sigma = sigmaIdx.request();
            auto info_r = r.request();
            auto info_mask = masks.request();
            compressor.encode_sgm((const uint8_t*)info_sigma.ptr, (int16_t*)info_r.ptr, (const bool*)info_mask.ptr, info_sigma.size);
        })
        .def("encode_factorize", [] (ANSEncoder &compressor, const py::buffer &cdfs, py::buffer &z) {
            auto infoCDF = cdfs.request();
            auto infoZ = z.request();
            compressor.encode_factorize((const uint8_t*)infoCDF.ptr, (uint8_t*)infoZ.ptr, infoCDF.shape[0], infoZ.shape[1]);
        })
        .def("get_memory", [] (ANSEncoder &compressor, py::buffer &memory) {
            auto infoMemory = memory.request();
            auto *memoryPtr = compressor.get_memory_ptr();
            memcpy(infoMemory.ptr, memoryPtr, infoMemory.size * sizeof(uint8_t));
        })
        .def("get_thread_sizes", [] (ANSEncoder &compressor, py::buffer &thread_sizes) {
            auto infoThreads = thread_sizes.request();
            auto *threadsPtr = compressor.get_thread_sizes_ptr();
            memcpy(infoThreads.ptr, threadsPtr, infoThreads.size * sizeof(uint32_t));
        })
        .def("get_z_transactions", [] (ANSEncoder &compressor, const py::buffer &cdfs, py::buffer &transactions) {
            auto infoCDF = cdfs.request();
            auto infoT = transactions.request();
            compressor.get_z_transactions((const uint8_t*)infoCDF.ptr, (uint32_t*)infoT.ptr);
        })
        .def("close", &ANSEncoder::close)
    ;

    py::class_<ANSDecoder>(m, "ANSDecoder")
        .def(py::init([] (py::buffer &memory, py::buffer &offsets) {
            auto infoMemory = memory.request();
            auto infoOffsets = offsets.request();
            return std::unique_ptr<ANSDecoder>(new ANSDecoder(
                (const uint8_t*)infoMemory.ptr, infoMemory.size, (const uint32_t*)infoOffsets.ptr, infoOffsets.size));
        }))
        .def("set_sgm_transitions", [] (ANSDecoder &decompressor, const py::buffer &transitions, const py::buffer &bounds) {
            auto infoTransition = transitions.request();
            auto infoBound = bounds.request();
            decompressor.set_sgm_transitions((const uint32_t*)infoTransition.ptr, (const uint8_t*)infoBound.ptr, infoTransition.shape[0]);
        })
        .def("decode_sgm", [] (ANSDecoder &decompressor, const py::buffer &sigmaIdx, py::buffer &r, const py::buffer &masks) {
            auto infoSigma = sigmaIdx.request();
            auto infoR = r.request();
            auto infoMasks = masks.request();
            decompressor.decode_sgm((const uint8_t*)infoSigma.ptr, (int16_t*)infoR.ptr, (const bool*)infoMasks.ptr, infoR.size);
        })
        .def("decode_factorize", [] (ANSDecoder &decompressor, const py::buffer &cdfs, py::buffer &z) {
            auto infoCDF = cdfs.request();
            auto infoZ = z.request();
            decompressor.decode_factorize((const uint8_t*)infoCDF.ptr, (uint8_t*)infoZ.ptr, infoCDF.shape[0], infoZ.shape[1]);
        })
    ;
}
