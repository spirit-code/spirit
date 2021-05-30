#pragma once
#ifndef LIBOVF_DETAIL_HELPERS_H
#define LIBOVF_DETAIL_HELPERS_H

#include <cstdint>

namespace ovf
{
namespace detail
{

// Test values for 4bit and 8bit binary data
namespace check
{
    static const uint32_t val_4b = 0x4996B438;
    static const uint64_t val_8b = 0x42DC12218377DE40;
}

namespace endian
{
    // union
    // {
    //     uint16_t s;
    //     unsigned char c[2];
    // } constexpr static  d {1};

    // constexpr bool is_little()
    // {
    //     return d.c[0] == 1;
    // }

    inline uint32_t from_little_32(const uint8_t * bytes)
    {
        return
            (uint32_t)bytes[0] <<  0 |
            (uint32_t)bytes[1] <<  8 |
            (uint32_t)bytes[2] << 16 |
            (uint32_t)bytes[3] << 24 ;
    }

    inline uint64_t from_little_64(const uint8_t * bytes)
    {
        return
            (uint64_t)bytes[0] <<  0 |
            (uint64_t)bytes[1] <<  8 |
            (uint64_t)bytes[2] << 16 |
            (uint64_t)bytes[3] << 24 |
            (uint64_t)bytes[4] << 32 |
            (uint64_t)bytes[5] << 40 |
            (uint64_t)bytes[6] << 48 |
            (uint64_t)bytes[7] << 56 ;
    }

    inline void to_little_32(const uint32_t & in, uint8_t * bytes)
    {
        bytes[0] = in >> 0;
        bytes[1] = in >> 8;
        bytes[2] = in >> 16;
        bytes[3] = in >> 24;
    }

    inline void to_little_64(const uint64_t & in, uint8_t * bytes)
    {
        bytes[0] = in >> 0;
        bytes[1] = in >> 8;
        bytes[2] = in >> 16;
        bytes[3] = in >> 24;
        bytes[4] = in >> 32;
        bytes[5] = in >> 40;
        bytes[6] = in >> 48;
        bytes[7] = in >> 56;
    }
}
}
}

#endif