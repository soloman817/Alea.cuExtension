//
// Copyright (c) 2011 QuantAlea GmbH.  All rights reserved.
// 

#include <algorithm>

#include "vector256.hpp"

namespace alea { namespace cuda { namespace math { namespace f2 {

    Vector256::Vector256(void)
    {
        // Initialize all bits to 0.
        std::fill(bits_, bits_ + 8, 0);
    }

    Vector256::Vector256(const unsigned* bits)
    {
        // Initialize all bits to given values.
        std::copy(bits, bits + 8, bits_);
    }

    bool Vector256::operator==(const Vector256& v) const
    {
        // Test for equality.
        for (int i = 0; i < 8; ++i)
            if (bits_[i] != v.bits_[i])
                return false;
        return true;
    }

    Vector256 Vector256::operator+(const Vector256& v) const
    {
        Vector256 r;

        // Calculate bitwise sum, in modulo-2 bit arithmetic.
        for (int i = 0; i < 8; ++i)
            r.bits_[i] = bits_[i] ^ v.bits_[i];

        return r;
    }
    
    unsigned Vector256::operator*(const Vector256& v) const
    {
        unsigned r = 0;

        // Calculate partial sums of products of corresponding bits of
        // given operands.  There will be 32 partial sums calculated,
        // represented by bits of an 32-bit unsigned number, where bit i
        // of this number will represent sum of bits i, i+32, i+64,
        // etc. of given operands.
        unsigned p = 0;
        for (int i = 0; i < 8; ++i)
            p ^= bits_[i] & v.bits_[i];

        // Add partial sums together.  First, upper and lower 16 partial
        // sums are summed together, producing 16 partials sums; then
        // upper and lower 8 of these 16 partial sums are summed
        // together, etc.
        r = p;
        r = (r >> 16) ^ (r & 0xffff);
        r = (r >> 8) ^ (r & 0xff);
        r = (r >> 4) ^ (r & 0xf);
        r = (r >> 2) ^ (r & 0x3);
        r = (r >> 1) ^ (r & 0x1);
        
        return r;
    }
    
}}}}

