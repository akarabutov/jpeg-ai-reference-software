#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <cstdint>
const int NUM_DISTRIBUTIONS_R = 32;
const int MAX_Z = 63;

const unsigned BIT_MASK[] = {
    0,          1,         3,         7,         0xF,       0x1F,
    0x3F,       0x7F,      0xFF,      0x1FF,     0x3FF,     0x7FF,
    0xFFF,      0x1FFF,    0x3FFF,    0x7FFF,    0xFFFF,    0x1FFFF,
    0x3FFFF,    0x7FFFF,   0xFFFFF,   0x1FFFFF,  0x3FFFFF,  0x7FFFFF,
    0xFFFFFF,   0x1FFFFFF, 0x3FFFFFF, 0x7FFFFFF, 0xFFFFFFF, 0x1FFFFFFF,
    0x3FFFFFFF, 0x7FFFFFFF};

const uint64_t BIT_MASK_HIGH[] = {
    0xffffffffffffff, 0x1ffffffffffffff, 0x3ffffffffffffff, 0x7ffffffffffffff, 
    0xfffffffffffffff, 0x1fffffffffffffff, 0x3fffffffffffffff, 0x7fffffffffffffff
};

#endif
