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
#include <algorithm>

#include "compressor.h"


BitStreamEncode::BitStreamEncode() 
{

}

BitStreamEncode::BitStreamEncode(uint8_t *ptr) : ptrEnd(ptr) 
{

}

void BitStreamEncode::init(uint8_t *ptr) 
{
    ptrEnd = ptr;
    head = 0;
    bitPos = 0;
    state1 = 0;
    state2 = 0;
}

void BitStreamEncode::write(uint64_t n, unsigned step) 
{
    head |= n << bitPos;
    bitPos += step;
}

void BitStreamEncode::flush() 
{
    *(uint64_t*)ptrEnd = head;
    ptrEnd += bitPos >> 3;
    head >>= bitPos & -8;
    bitPos &= 7;
}

uint8_t* BitStreamEncode::close() 
{
    *(uint64_t*)ptrEnd = head;
    return ptrEnd + ((bitPos + 7) >> 3);
}

uint8_t* BitStreamEncode::closeWithState() 
{
    uint64_t pushed = 256 | state1;
    write(pushed, 9);
    return close();
}

uint8_t* BitStreamEncode::closeWithStates() 
{
    uint64_t pushed = 65536 | (state1 << 8) | state2;
    write(pushed, 17);
    return close();
}


ANSEncoder::ANSEncoder(ssize_t capacity, int nThreads_) : nThreads(nThreads_)
{
    auto capacityEachThread = capacity / nThreads;
    this->memory = new uint8_t[capacity];
    this->threadSizes = new uint32_t[nThreads_];
    this->ptrStarts = new uint8_t*[nThreads];
    this->streams = new BitStreamEncode[nThreads];

    for (int i = 0; i < nThreads; ++i) 
    {
        this->threadSizes[i] = 0;
        this->ptrStarts[i] = this->memory + capacityEachThread * i;
        this->streams[i].init(ptrStarts[i]);
    }

    int nBits = 1;
    for (int i = 255; i; --i) {
        if (i << nBits < 256) {
            ++nBits;
        }
        this->deltaBits[i] = (nBits << 8) - (i << nBits) + 256;
    }
}

ANSEncoder::~ANSEncoder()
{
    delete [] this->memory;
    delete [] this->threadSizes;
    delete [] this->ptrStarts;
    delete [] this->streams;
    
    if (this->transitionsYAll != nullptr) {
        delete [] transitionsYAll;
    }

    if (this->stateMapsYAll != nullptr) {
        delete [] stateMapsYAll;
    }
}


void ANSEncoder::get_z_transactions(const uint8_t *cdfs, uint32_t* transactions)
{
    setZDistributions(cdfs);
    std::copy(transitionsZ, transitionsZ+256, transactions);
}

void ANSEncoder::encode_sgm(const uint8_t *sigmaIdx, int16_t *r, const bool *masks, ssize_t size)
{
    if (nThreads == 1) {
        encodeRSingleThread(sigmaIdx, r, masks, size);
    } else {
        encodeRMultiThread(sigmaIdx, r, masks, size);
    }
}

void ANSEncoder::encodeRSingleThread(const uint8_t *sigmaIdx, int16_t *r, const bool *masks, ssize_t index) {
    BitStreamEncode stream;
    memcpy(&stream, &streams[0], sizeof(stream));

    if (index & 1) {
        index--;
        if (masks[index]) {
            encodeYOutbound(stream, sigmaIdx, r, index);
            encodeYInbound(stream, stream.state1, sigmaIdx, r, index);
        }
		stream.flush();
    }

    if (index & 2) {
        index -= 2;
        if (masks[index + 1]) {
            encodeYOutbound(stream, sigmaIdx, r, index + 1);
            encodeYInbound(stream, stream.state2, sigmaIdx, r, index + 1);
        }
        if (masks[index]) {
            encodeYOutbound(stream, sigmaIdx, r, index);
            encodeYInbound(stream, stream.state1, sigmaIdx, r, index);
        }
		stream.flush();
    }

    while (index) {
        index -= 4;
        if (masks[index + 3]) encodeYOutbound(stream, sigmaIdx, r, index + 3);
        if (masks[index + 2]) encodeYOutbound(stream, sigmaIdx, r, index + 2);
        if (masks[index + 1]) encodeYOutbound(stream, sigmaIdx, r, index + 1);
        if (masks[index]) encodeYOutbound(stream, sigmaIdx, r, index);
        if (masks[index + 3]) encodeYInbound(stream, stream.state2, sigmaIdx, r, index + 3);
        if (masks[index + 2]) encodeYInbound(stream, stream.state1, sigmaIdx, r, index + 2);
        if (masks[index + 1]) encodeYInbound(stream, stream.state2, sigmaIdx, r, index + 1);
        if (masks[index]) encodeYInbound(stream, stream.state1, sigmaIdx, r, index);
        stream.flush();
    }

    memcpy(&streams[0], &stream, sizeof(stream));
}


void ANSEncoder::encode_factorize(const uint8_t *cdfs, uint8_t *z, int channels, ssize_t sizePerChannel)
{
    if (nThreads == 1) {
        for (int i = channels - 1; i >= 0; --i) {
            encodeZRowSingleThread(cdfs + i * MAX_Z, z + i * sizePerChannel, sizePerChannel);
        }
    } else {
        for (int i = channels - 1; i >= 0; --i) {
            encodeZRowMultiThread(cdfs + i * MAX_Z, z + i * sizePerChannel, sizePerChannel);
        }
    }
}

void ANSEncoder::encodeZRowSingleThread(const uint8_t *cdfs, uint8_t *z, ssize_t length) 
{
    BitStreamEncode stream;
    memcpy(&stream, streams, sizeof(stream));

    setZDistributions(cdfs);
    if (length & 1) {
        encodeZOutbound(stream, z, --length);
        encodeZInbound(stream, stream.state1, z, length);
    }
    if (length & 2) {
        encodeZOutbound(stream, z, --length);
        encodeZInbound(stream, stream.state2, z, length);
        encodeZOutbound(stream, z, --length);
        encodeZInbound(stream, stream.state1, z, length);
    }
    stream.flush();
    while (length) {
        length -= 4;
        encodeZOutbound(stream, z, length + 3);
        encodeZOutbound(stream, z, length + 2);
        encodeZOutbound(stream, z, length + 1);
        encodeZOutbound(stream, z, length);
        encodeZInbound(stream, stream.state2, z, length + 3);
        encodeZInbound(stream, stream.state1, z, length + 2);
        encodeZInbound(stream, stream.state2, z, length + 1);
        encodeZInbound(stream, stream.state1, z, length);
        stream.flush();
    }

    memcpy(streams, &stream, sizeof(stream));
}


const uint32_t* ANSEncoder::get_thread_sizes_ptr() const
{
    return this->threadSizes;
}

const uint8_t* ANSEncoder::get_memory_ptr() const
{
    return this->memory;
}

uint32_t ANSEncoder::close() 
{
    if (this->nThreads == 1) {
        this->threadSizes[0] = this->streams[0].closeWithStates() - this->ptrStarts[0];
        return this->threadSizes[0];
    } else {
        uint8_t *ptrCurr = this->ptrStarts[0];
        for (int i = 0; i < this->nThreads; ++i) {
            this->threadSizes[i] = this->streams[i].closeWithState() - this->ptrStarts[i];
            if (i) {
                memmove(ptrCurr, this->ptrStarts[i], this->threadSizes[i]);
            }
            ptrCurr += this->threadSizes[i];
        }
        return ptrCurr - this->ptrStarts[0];
    }
}

void ANSEncoder::set_sgm_transitions(const uint32_t *transitions, const uint8_t *bounds, const uint8_t* stateMaps, ssize_t numDistributions, ssize_t dimY)
{
    this->transitionsYAll = new uint32_t[numDistributions * dimY];
    this->stateMapsYAll = new uint8_t[numDistributions * 256];
    memcpy(this->transitionsYAll, transitions, numDistributions * dimY * sizeof(uint32_t));
    memcpy(this->stateMapsYAll, stateMaps, numDistributions * 256 * sizeof(uint8_t));

    for (int i = 0; i < NUM_DISTRIBUTIONS_R; ++i) {
        this->transitionsY[i] = this->transitionsYAll + i * dimY + dimY / 2;
        this->stateMapsY[i] = this->stateMapsYAll + i * 256;
    }
    memcpy(boundsY, bounds, NUM_DISTRIBUTIONS_R);
}

void ANSEncoder::encodeRMultiThread(const uint8_t *sigmaIdx, int16_t *r, const bool *masks, ssize_t length) 
{
    int numRounds = length / nThreads;
    int numGroups = numRounds >> 2;
    int roundThres = numRounds * nThreads, groupThres = (numGroups * nThreads) << 2;

    #pragma omp parallel num_threads(nThreads)
    {
        int i = omp_get_thread_num();
        BitStreamEncode subStream;
        memcpy(&subStream, streams + i, sizeof(subStream));
        int index = roundThres + i - nThreads * (i + roundThres >= length);

        while (index >= groupThres) {
            if (masks[index]) {
                encodeYOutbound(subStream, sigmaIdx, r, index);
                encodeYInbound(subStream, (nThreads>1 || ((index & 0x1)==0)) ? subStream.state1: subStream.state2, sigmaIdx, r, index);
                subStream.flush();
            }
            index -= nThreads;
        }
        
        while (index > 0) {
            if (masks[index]) encodeYOutbound(subStream, sigmaIdx, r, index);
            if (masks[index - nThreads]) encodeYOutbound(subStream, sigmaIdx, r, index - nThreads);
            if (masks[index - nThreads * 2]) encodeYOutbound(subStream, sigmaIdx, r, index - nThreads * 2);
            if (masks[index - nThreads * 3]) encodeYOutbound(subStream, sigmaIdx, r, index - nThreads * 3);
            if (masks[index]) encodeYInbound(subStream, (nThreads>1)?subStream.state1:subStream.state2, sigmaIdx, r, index);
            if (masks[index - nThreads]) encodeYInbound(subStream, subStream.state1, sigmaIdx, r, index - nThreads);
            if (masks[index - nThreads * 2]) encodeYInbound(subStream, (nThreads>1)?subStream.state1:subStream.state2, sigmaIdx, r, index - nThreads * 2);
            if (masks[index - nThreads * 3]) encodeYInbound(subStream, subStream.state1, sigmaIdx, r, index - nThreads * 3);
            index -= nThreads * 4;
            subStream.flush();
        }

        memcpy(streams + i, &subStream, sizeof(subStream));
    }
}

void ANSEncoder::encodeYInbound(BitStreamEncode &stream, uint8_t &state, const uint8_t *sigmaIdx, const int16_t *r, ssize_t index) 
{
    encodeWithTransition(stream, state, transitionsY[sigmaIdx[index]][r[index]]);
    state = stateMapsY[sigmaIdx[index]][state];
}

void ANSEncoder::encodeYOutbound(BitStreamEncode &stream, const uint8_t *sigmaIdx, int16_t *r, ssize_t index) 
{
    int value = r[index];
    int bound = boundsY[sigmaIdx[index]];
    if (value >= bound || value <= -bound) {
        const int32_t sign = value >> 15;
        uint64_t valueCoded = ((value + (sign ^ (-bound))) << 1) ^ sign;
        stream.head |= valueCoded << stream.bitPos;
        if (valueCoded >= 8) {
            stream.bitPos += 17;
        } else{
            stream.head |= 1LU << (stream.bitPos + 3);
            stream.bitPos += 4;
        }
        stream.flush();
        r[index] = -bound;
    }
}

void ANSEncoder::encodeWithTransition(BitStreamEncode &stream, uint8_t &state, uint32_t transition) {
    uint32_t nBits = (state + (transition >> 16)) >> 8;
    stream.write(state & BIT_MASK[nBits], nBits);
    state = ((state | 256) >> nBits) + transition;
}

void ANSEncoder::encodeZRowMultiThread(const uint8_t *cdfs, uint8_t *z, ssize_t length) 
{
    setZDistributions(cdfs);
    int numRounds = length / nThreads;
    int numGroups = numRounds >> 2;
    int roundThres = numRounds * nThreads, groupThres = (numGroups * nThreads) << 2;
    #pragma omp parallel num_threads(nThreads)
    {
        int i = omp_get_thread_num();
        BitStreamEncode subStream;
        memcpy(&subStream, streams + i, sizeof(subStream));
        int index = roundThres + i - nThreads * (i + roundThres >= length);
        while (index >= groupThres) {
            encodeZOutbound(subStream, z, index);
            encodeZInbound(subStream, (nThreads>1 || ((index & 0x1)==0))?subStream.state1:subStream.state2, z, index);
            index -= nThreads;
        }
        subStream.flush();
        while (index > 0) {
            encodeZOutbound(subStream, z, index);
            encodeZOutbound(subStream, z, index - nThreads);
            encodeZOutbound(subStream, z, index - nThreads * 2);
            encodeZOutbound(subStream, z, index - nThreads * 3);
            encodeZInbound(subStream, (nThreads>1)?subStream.state1:subStream.state2, z, index);
            encodeZInbound(subStream, subStream.state1, z, index - nThreads);
            encodeZInbound(subStream, (nThreads>1)?subStream.state1:subStream.state2, z, index - nThreads * 2);
            encodeZInbound(subStream, subStream.state1, z, index - nThreads * 3);
            index -= nThreads * 4;
            subStream.flush();
        }

        memcpy(streams + i, &subStream, sizeof(subStream));
    }
}

void ANSEncoder::setZDistributions(const uint8_t *cdfs) 
{
    int cdfCurr = 0;
    for (int i = 0; i < MAX_Z; ++i) {
        int pmf = cdfs[i] - cdfCurr;
        transitionsZ[i] = pmf ? (deltaBits[pmf] << 16) | uint8_t(cdfCurr - pmf) : 0;
        cdfCurr = cdfs[i];
    }
    transitionsZ[MAX_Z] = (deltaBits[1] << 16) | 254;
}

void ANSEncoder::encodeZInbound(BitStreamEncode &stream, uint8_t &state, uint8_t *values, ssize_t index) 
{
    encodeWithTransition(stream, state, transitionsZ[values[index]]);
}

void ANSEncoder::encodeZOutbound(BitStreamEncode &stream, uint8_t *values, ssize_t index) {
    if (!transitionsZ[values[index]]) {
        stream.write(values[index], 6);
        values[index] = MAX_Z;
    }
}
