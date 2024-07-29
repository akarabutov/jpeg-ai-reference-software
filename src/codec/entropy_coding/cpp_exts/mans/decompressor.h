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
