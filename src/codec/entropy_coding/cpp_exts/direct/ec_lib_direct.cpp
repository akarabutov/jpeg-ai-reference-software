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

#include <cstring>
#include "ec_lib_direct.h"

EcLibDirect::EcLibDirect(uint8_t *data, size_t data_size) : m_data(data), m_data_size(data_size), m_cur_position(0)
{

}

EcLibDirect::~EcLibDirect()
{

}

uint8_t  EcLibDirect::read_bits(uint32_t& data, uint8_t bits_count)
{
    int byte_pos = m_cur_position >> 3;
    int bit_idx = m_cur_position & 0x7;
    uint8_t output_count = 0;
    data = 0;
    while (bits_count-- && (byte_pos < m_data_size))
    {
        uint8_t cur_bit_idx = 7 - bit_idx;
        uint8_t bit = (m_data[byte_pos] >> cur_bit_idx) & 0x1;
        data <<= 1;
        data |= bit;
        ++output_count;
        ++bit_idx;
        if (bit_idx == 8)
        {
            ++byte_pos;
            bit_idx = 0;
        }
    }
    m_cur_position = (byte_pos << 3) + bit_idx;
    return output_count;
}

void EcLibDirect::write_bits(int data, uint8_t bits_count)
{
    int out_byte_pos = m_cur_position >> 3;
    int out_bit_idx = m_cur_position & 0x7;
    while (bits_count--)
    {
        uint8_t cur_bit_idx = 7 - out_bit_idx;
        uint8_t bit = (data >> bits_count) & 0x1;
        m_data[out_byte_pos] |= bit << cur_bit_idx;
        ++out_bit_idx;
        if (out_bit_idx == 8)
        {
            ++out_byte_pos;
            out_bit_idx = 0;
        }
    }
    m_cur_position = (out_byte_pos << 3) + out_bit_idx;
    
}

void   EcLibDirect::clear()
{
    memset(m_data, 0x00, m_data_size);
    m_cur_position = 0;
}

void   EcLibDirect::set_pointer(size_t bit_position)
{
    m_cur_position = bit_position;
}

size_t EcLibDirect::get_pointer_pos() const
{
    return m_cur_position;
}
