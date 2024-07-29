#pragma once
#include <cstddef>
#include <cstdint>

class EcLibDirect
{
public:
    EcLibDirect(uint8_t* data, size_t data_size);
    ~EcLibDirect();

    uint8_t         read_bits(uint32_t& data, uint8_t bits_count);
    void            write_bits(int data, uint8_t bits_count);

    void            set_pointer(size_t bit_position);
    size_t          get_pointer_pos() const;

    //! Clear all data and pointer
    void            clear();

protected:
    uint8_t*        m_data;                 //!< Reference on a memory with data. The object doesn't control for removing of the memory.
    size_t          m_data_size;            //!< Length/max-length of m_data (in bytes)

    size_t          m_cur_position;         //!< Current position in the memory (in bits)
};