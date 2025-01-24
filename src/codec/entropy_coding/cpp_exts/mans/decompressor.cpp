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

#include <omp.h>
#include "decompressor.h"


void BitStreamDecode::initHeader(const uint8_t *ptr) {
    ptrEnd = ptr;
    head = *(uint64_t*) ptrEnd;
    bitPos = 0;
}

uint64_t BitStreamDecode::read(unsigned step) {
    bitPos -= step;
    uint64_t result = head >> bitPos;
    head ^= result << bitPos;
    return result;
}

void BitStreamDecode::flush() {
    ptrEnd -= 7 ^ (bitPos >> 3);
    head = *(uint64_t*)ptrEnd & BIT_MASK_HIGH[bitPos & 7];
    bitPos |= 56;
}

void BitStreamDecode::initBackward(const uint8_t *ptr) {
    ptrEnd = ptr - 8;
    head = *(uint64_t*) ptrEnd;
    bitPos = 63 ^ __builtin_clzll(head);
    head ^= 1LU << bitPos;
}

void BitStreamDecode::initWithState(const uint8_t *ptr) {
    initBackward(ptr);
    state1 = read(8);
    flush();
}

void BitStreamDecode::initWithStates(const uint8_t *ptr) {
    initBackward(ptr);
    state1 = read(8);
    state2 = read(8);
    flush();
}

ANSDecoder::ANSDecoder(const uint8_t *ptr, ssize_t capacity, const uint32_t *threadOffsets, int nThreads_): nThreads(nThreads_) 
{
    this->memory = new uint8_t[capacity + 8];
    memcpy(this->memory + 8, ptr, capacity * sizeof(uint8_t));

    this->threadOffsets = new uint32_t[nThreads_];
    for (int i = 0; i < nThreads_; ++i) {
        this->threadOffsets[i] = threadOffsets[i] + 8;
    }

    this->streams = new BitStreamDecode[nThreads];
    if (nThreads == 1) {
        this->streams[0].initWithStates(this->memory + this->threadOffsets[0]);
    } else {
        for (int i = 0; i < nThreads; ++i) {
            this->streams[i].initWithState(this->memory + this->threadOffsets[i]);
        }
    }

    int nBits = 0;
    for (int i = 511; i; --i) {
        if (i << nBits < 256) {
            ++nBits;
        }
        transitionsZPreprocess[i] = ((i << nBits) ^ 256) | (nBits << 8);
    }
}

ANSDecoder::~ANSDecoder() {
    delete [] this->memory;
    delete [] this->threadOffsets;
    delete [] this->streams;

    if (this->transitionsYAll != nullptr) {
        delete [] this->transitionsYAll;
    }
}

void ANSDecoder::set_sgm_transitions(const uint32_t *transitions, const uint8_t *bounds, ssize_t numDistributions) {
    this->transitionsYAll = new uint32_t[numDistributions * 256];
    memcpy(this->transitionsYAll, transitions, numDistributions * 256 * sizeof(uint32_t));
    memcpy(boundsY, bounds, numDistributions * sizeof(uint8_t));

    for (int i = 0; i < numDistributions; ++i) {
        this->transitionsY[i] = this->transitionsYAll + 256 * i;
    }
}

void ANSDecoder::decode_sgm(const uint8_t *sigmaIdx, int16_t *r, const bool *masks, ssize_t size)
{
    if (nThreads == 1) {
        decodeSingleThread(sigmaIdx, r, masks, size);
    } else {
        decodeMultiThread(sigmaIdx, r, masks, size);
    }
}


void ANSDecoder::decodeSingleThread(const uint8_t *sigmaIdx, int16_t *r, const bool *masks, ssize_t length) {
    BitStreamDecode stream;
    memcpy(&stream, streams, sizeof(stream));
    ssize_t index = 0;

    while (index + 3 < length) {
        if (masks[index]) decodeYInbound(stream, stream.state1, sigmaIdx, r, index);
        if (masks[index + 1]) decodeYInbound(stream, stream.state2, sigmaIdx, r, index + 1);
        if (masks[index + 2]) decodeYInbound(stream, stream.state1, sigmaIdx, r, index + 2);
        if (masks[index + 3]) decodeYInbound(stream, stream.state2, sigmaIdx, r, index + 3);
        stream.flush();
        if (masks[index]) decodeYOutbound(stream, sigmaIdx, r, index);
        if (masks[index + 1]) decodeYOutbound(stream, sigmaIdx, r, index + 1);
        if (masks[index + 2]) decodeYOutbound(stream, sigmaIdx, r, index + 2);
        if (masks[index + 3]) decodeYOutbound(stream, sigmaIdx, r, index + 3);
        index += 4;
    }

    if (length & 2) {
        if (masks[index]) {
            decodeYInbound(stream, stream.state1, sigmaIdx, r, index);
            decodeYOutbound(stream, sigmaIdx, r, index);
        }
        if (masks[index + 1]) {
            decodeYInbound(stream, stream.state2, sigmaIdx, r, index + 1);
            decodeYOutbound(stream, sigmaIdx, r, index + 1);
        }
        index += 2;
		stream.flush();
    }

    if (length & 1) {
        if (masks[index]) {
            decodeYInbound(stream, stream.state1, sigmaIdx, r, index);
            decodeYOutbound(stream, sigmaIdx, r, index);
        }
		stream.flush();
    }

    memcpy(streams, &stream, sizeof(stream));
}


void ANSDecoder::decode_factorize(const uint8_t *cdfs, uint8_t *z, ssize_t channels, ssize_t sizePerChannel)
{
    if (nThreads == 1) {
        for (int i = 0; i < channels; ++i) {
            decodeRowSingleThread(cdfs + i * MAX_Z, z + i * sizePerChannel, sizePerChannel);
        }
    } else {
        for (int i = 0; i < channels; ++i) {
            decodeRowMultiThread(cdfs + i * MAX_Z, z + i * sizePerChannel, sizePerChannel);
        }
    }
}

void ANSDecoder::decodeRowSingleThread(const uint8_t *cdfs, uint8_t *values, ssize_t length) {
    BitStreamDecode stream;
    memcpy(&stream, streams, sizeof(stream));
    setZDistributions(cdfs);
    int index = 0;
    while (index + 3 < length) {
        decodeZInbound(stream, stream.state1, values, index);
        decodeZInbound(stream, stream.state2, values, index + 1);
        decodeZInbound(stream, stream.state1, values, index + 2);
        decodeZInbound(stream, stream.state2, values, index + 3);
        decodeZOutbound(stream, values, index);
        decodeZOutbound(stream, values, index + 1);
        decodeZOutbound(stream, values, index + 2);
        decodeZOutbound(stream, values, index + 3);
        stream.flush();
        index += 4;
    }

    if (length & 2) {
        decodeZInbound(stream, stream.state1, values, index);
        decodeZOutbound(stream, values, index);
        decodeZInbound(stream, stream.state2, values, index + 1);
        decodeZOutbound(stream, values, index + 1);
        index += 2;
    }

    if (length & 1) {
        decodeZInbound(stream, stream.state1, values, index);
        decodeZOutbound(stream, values, index);
    }
    stream.flush();
    memcpy(streams, &stream, sizeof(stream));
}

void ANSDecoder::decodeMultiThread(const uint8_t *sigmaIdx, int16_t *r, const bool *masks, ssize_t length) {
    unsigned thres = length / (nThreads << 2) * (nThreads << 2);
    #pragma omp parallel num_threads(nThreads)
    {
        unsigned i = omp_get_thread_num();
        unsigned index = i;
        BitStreamDecode subStream;
        memcpy(&subStream, streams + i, sizeof(subStream));
        
        while (index < thres) {
            if (masks[index]) decodeYInbound(subStream, subStream.state1, sigmaIdx, r, index);
            if (masks[index + nThreads]) decodeYInbound(subStream, (nThreads>1)?subStream.state1:subStream.state2, sigmaIdx, r, index + nThreads);
            if (masks[index + nThreads * 2]) decodeYInbound(subStream, subStream.state1, sigmaIdx, r, index + nThreads * 2);
            if (masks[index + nThreads * 3]) decodeYInbound(subStream, (nThreads>1)?subStream.state1:subStream.state2, sigmaIdx, r, index + nThreads * 3);
            subStream.flush();
            if (masks[index]) decodeYOutbound(subStream, sigmaIdx, r, index);
            if (masks[index + nThreads]) decodeYOutbound(subStream, sigmaIdx, r, index + nThreads);
            if (masks[index + nThreads * 2]) decodeYOutbound(subStream, sigmaIdx, r, index + nThreads * 2);
            if (masks[index + nThreads * 3]) decodeYOutbound(subStream, sigmaIdx, r, index + nThreads * 3);
            index += nThreads * 4;
        }
        while (index < length) {
            if (masks[index]) decodeYInbound(subStream, (nThreads>1 || ((index & 0x1)==0))?subStream.state1:subStream.state2, sigmaIdx, r, index);
            if (masks[index]) decodeYOutbound(subStream, sigmaIdx, r, index);
            index += nThreads;
			subStream.flush();
        }

        memcpy(streams + i, &subStream, sizeof(subStream));
    }
}

void ANSDecoder::decodeYInbound(BitStreamDecode &stream, uint8_t &state, const uint8_t *stds, int16_t *shifts, ssize_t index) {
    uint32_t transition = transitionsY[stds[index]][state];
    state = (transition >> 16) | stream.read(transition >> 24);
    shifts[index] = transition;
}

void ANSDecoder::decodeYOutbound(BitStreamDecode &stream, const uint8_t *stds, int16_t *shifts, ssize_t index) {
    int bound = boundsY[stds[index]];
    if (!(shifts[index] + bound)) {
        auto value_temp = stream.read(stream.read(1) ? 3 : 16);
        int sign = value_temp & 1;
        shifts[index] = ((value_temp >> 1) + (bound - sign)) ^ (-sign);
        stream.flush();
    }
}

void ANSDecoder::decodeRowMultiThread(const uint8_t *cdfs, uint8_t *values, ssize_t length) {
    setZDistributions(cdfs);
    unsigned thres = length / (nThreads << 2) * (nThreads << 2);
    #pragma omp parallel num_threads(nThreads)
    {
        unsigned i = omp_get_thread_num();
        unsigned index = i;
        BitStreamDecode subStream;
        memcpy(&subStream, streams + i, sizeof(subStream));

        while (index < thres) {
            decodeZInbound(subStream, subStream.state1, values, index);
            decodeZInbound(subStream, (nThreads>1)?subStream.state1:subStream.state2, values, index + nThreads);
            decodeZInbound(subStream, subStream.state1, values, index + nThreads * 2);
            decodeZInbound(subStream, (nThreads>1)?subStream.state1:subStream.state2, values, index + nThreads * 3);
            decodeZOutbound(subStream, values, index);
            decodeZOutbound(subStream, values, index + nThreads);
            decodeZOutbound(subStream, values, index + nThreads * 2);
            decodeZOutbound(subStream, values, index + nThreads * 3);
            subStream.flush();
            index += nThreads * 4;
        }
        while (index < length) {
            decodeZInbound(subStream, (nThreads>1 || ((index & 0x1)==0))?subStream.state1:subStream.state2, values, index);
            decodeZOutbound(subStream, values, index);
            index += nThreads;
        }
        subStream.flush();

        memcpy(streams + i, &subStream, sizeof(subStream));
    }
}

void ANSDecoder::setZDistributions(const uint8_t *cdfs) {
    int curr = 0;
    for (int i = 0; i < MAX_Z; ++i) {
        int k = cdfs[i] - curr;
        for (unsigned j = curr; j < cdfs[i]; ++j) {
            transitionsZ[j] = (transitionsZPreprocess[k++] << 16) | i;
        }
        curr = cdfs[i];
    }
    transitionsZ[255] = (transitionsZPreprocess[1] << 16) | MAX_Z;
}

void ANSDecoder::decodeZInbound(BitStreamDecode &stream, uint8_t &state, uint8_t *values, ssize_t index) {
    uint32_t transition = transitionsZ[state];
    state = (transition >> 16) | stream.read(transition >> 24);
    values[index] = transition;
}

void ANSDecoder::decodeZOutbound(BitStreamDecode &stream, uint8_t *values, ssize_t index) {
    if (values[index] == MAX_Z) {
        values[index] = stream.read(6);
    }
}
