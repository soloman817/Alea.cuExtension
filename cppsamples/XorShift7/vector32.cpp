//
// Copyright (c) 2011 QuantAlea GmbH.  All rights reserved.
// 

#include "vector32.hpp"

namespace alea { namespace cuda { namespace math { namespace f2 {

    Vector32::Vector32(void)
        : bits_(0)
    {
    }

    Vector32::Vector32(const unsigned bits)
        : bits_(bits)
    {
    }

    bool Vector32::operator==(const Vector32& v) const
    {
        // Test for equality.
        return bits_ == v.bits_;
    }

    Vector32 Vector32::operator+(const Vector32& v) const
    {
        // Calculate bitwise sum, in modulo-2 bit arithmetic.
        return Vector32(bits_ ^ v.bits_);
    }

    unsigned Vector32::operator*(const Vector32& v) const
    {
        unsigned r = 0;

        // Calculate bitwise product, in modulo-2 bit arithmetic.
        unsigned p = bits_ & v.bits_;

        // Calculate sum, in modulo-2 bit arithmetic, of individual bits
        // of above product.  First, upper and lower 16 bits are summed
        // bitwise, producing 16 partial sums.  Then upper and lower 8
        // of these 16 partial sums are added together, producing 8
        // partial sums, etc.
        r = p;
        r = (r >> 16) ^ (r & 0xffff);
        r = (r >> 8) ^ (r & 0xff);
        r = (r >> 4) ^ (r & 0xf);
        r = (r >> 2) ^ (r & 0x3);
        r = (r >> 1) ^ (r & 0x1);

        return r;
    }

}}}}

