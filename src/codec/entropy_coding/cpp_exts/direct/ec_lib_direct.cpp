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
