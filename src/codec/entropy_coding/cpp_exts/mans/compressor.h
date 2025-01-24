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

#ifndef COMPRESSOR_H
#define COMPRESSOR_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "constants.h"

struct BitStreamEncode {
    uint8_t *ptrEnd{nullptr};
    uint64_t head{0};
    unsigned bitPos{0};
    uint8_t state1{0}, state2{0};

    BitStreamEncode();
    BitStreamEncode(uint8_t *ptr);

    void init(uint8_t *ptr);
    void write(uint64_t n, unsigned step);
    void flush();
    uint8_t* close();
    uint8_t* closeWithState();
    uint8_t* closeWithStates();
};

class ANSEncoder {
private:
    // Stream info
    uint8_t *memory{nullptr};
    uint32_t *threadSizes{nullptr};
    uint8_t **ptrStarts{nullptr};
    uint32_t *transitionsYAll{nullptr};
    uint8_t *stateMapsYAll{nullptr};

    BitStreamEncode *streams;
    int nThreads;

    // Transition tables for Y
    const uint32_t *transitionsY[NUM_DISTRIBUTIONS_R];
    const uint8_t *stateMapsY[NUM_DISTRIBUTIONS_R];
    uint8_t boundsY[NUM_DISTRIBUTIONS_R];

    // Transition tables for Z
    uint32_t transitionsZ[MAX_Z + 1];
    uint16_t deltaBits[256];

public:
    ANSEncoder(ssize_t capacity, int nThreads_);
    ~ANSEncoder();
    void encode_sgm(const uint8_t *sigmaIdx, int16_t *r, const bool *masks, ssize_t size);
    void encode_factorize(const uint8_t *cdfs, uint8_t *z, int channels, ssize_t sizePerChannel);
    void set_sgm_transitions(const uint32_t *transitions, const uint8_t *bounds, const uint8_t* stateMaps, ssize_t numDistributions, ssize_t dimY);
    const uint32_t* get_thread_sizes_ptr() const;
    const uint8_t* get_memory_ptr() const;
    uint32_t close();
    void get_z_transactions(const uint8_t *cdfs, uint32_t* transactions);

protected:
    void encodeRSingleThread(const uint8_t *sigmaIdx, int16_t *r, const bool *masks, ssize_t index);
    void encodeRMultiThread(const uint8_t *sigmaIdx, int16_t *r, const bool *masks, ssize_t length);
    void encodeYInbound(BitStreamEncode &stream, uint8_t &state, const uint8_t *sigmaIdx, const int16_t *r, ssize_t index);
    void encodeYOutbound(BitStreamEncode &stream, const uint8_t *sigmaIdx, int16_t *r, ssize_t index);
    void encodeZRowSingleThread(const uint8_t *cdfs, uint8_t *z, ssize_t length);
    void encodeZRowMultiThread(const uint8_t *cdfs, uint8_t *z, ssize_t length);
    void setZDistributions(const uint8_t *cdfs);
    void encodeZInbound(BitStreamEncode &stream, uint8_t &state, uint8_t *values, ssize_t index);
    void encodeZOutbound(BitStreamEncode &stream, uint8_t *values, ssize_t index);
    void encodeWithTransition(BitStreamEncode &stream, uint8_t &state, uint32_t transition);
};
#endif
