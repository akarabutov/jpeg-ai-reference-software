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

#ifndef DECOMPRESSOR_H
#define DECOMPRESSOR_H

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "constants.h"

struct BitStreamDecode 
{
    const uint8_t *ptrEnd{nullptr};
    uint64_t head{0};
    unsigned bitPos{0};
    uint8_t state1{0}, state2{0};

    void initHeader(const uint8_t *ptr);
    uint64_t read(unsigned step);
    unsigned readHeader(unsigned step);
    void flush();
    void initBackward(const uint8_t *ptr);
    void initWithState(const uint8_t *ptr);
    void initWithStates(const uint8_t *ptr);
};


class ANSDecoder 
{
private:
    // Stream info
    uint8_t *memory;
    uint32_t *threadOffsets;
    uint32_t *transitionsYAll;
    BitStreamDecode *streams;
    int nThreads;

    // Transition tables for R
    const uint32_t *transitionsY[NUM_DISTRIBUTIONS_R];
    uint8_t boundsY[NUM_DISTRIBUTIONS_R];

    // Transition tables for Z
    uint32_t transitionsZ[256] {0};
    uint16_t transitionsZPreprocess[512] {0};

public:
    ANSDecoder(const uint8_t *ptr, ssize_t capacity, const uint32_t *threadOffsets, int nThreads_);
    ~ANSDecoder();
    void set_sgm_transitions(const uint32_t *transitions, const uint8_t *bounds, ssize_t numDistributions);
    void decode_sgm(const uint8_t *sigmaIdx, int16_t *r, const bool *masks, ssize_t size);
    void decode_factorize(const uint8_t *cdfs, uint8_t *z, ssize_t channels, ssize_t sizePerChannel);

protected:
    void decodeSingleThread(const uint8_t *sigmaIdx, int16_t *r, const bool *masks, ssize_t length);
    void decodeMultiThread(const uint8_t *sigmaIdx, int16_t *r, const bool *masks, ssize_t length);
    void decodeYInbound(BitStreamDecode &stream, uint8_t &state, const uint8_t *stds, int16_t *shifts, ssize_t index);
    void decodeYOutbound(BitStreamDecode &stream, const uint8_t *stds, int16_t *shifts, ssize_t index);
    void decodeRowSingleThread(const uint8_t *cdfs, uint8_t *values, ssize_t length);
    void decodeRowMultiThread(const uint8_t *cdfs, uint8_t *values, ssize_t length);
    void setZDistributions(const uint8_t *cdfs);
    void decodeZInbound(BitStreamDecode &stream, uint8_t &state, uint8_t *values, ssize_t index);
    void decodeZOutbound(BitStreamDecode &stream, uint8_t *values, ssize_t index);
};


#endif
