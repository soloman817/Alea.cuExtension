//
// Copyright (c) 2011 QuantAlea GmbH.  All rights reserved.
// 

#include <algorithm>
#include <cassert>

#include "matrix32.hpp"

namespace alea { namespace cuda { namespace math { namespace f2 {

    Matrix32::Matrix32(void)
    {
        // Initialize all bits to 0.
        std::fill(bits_, bits_ + 32, 0);
    }

    Matrix32::Matrix32(const unsigned* bits)
    {
        // Initialize all bits to given values.
        std::copy(bits, bits + 32, bits_);
    }

    Matrix32 Matrix32::identity(void)
    {
        Matrix32 r;

        // Build identity matrix.
        unsigned value = (unsigned) (1 << 31);
        for (int i = 0; i < 32; ++i) {
            r.bits_[i] = value;
            value >>= 1;
        }

        return r;
    }
     
    Matrix32 Matrix32::left(const int n)
    {
        // Verify that function argument is valid.
        assert(n >=1 && n <= 31);

        Matrix32 r;

        // Build corresponding left-shift matrix.
        unsigned value = (unsigned) (1 << (31 - n));
        for (int i = 0; i < 32; ++i) {
            r.bits_[i] = value;
            value >>= 1;
        }

        return r;
    }
     
    Matrix32 Matrix32::right(const int n)
    {
        // Verify that function argument is valid.
        assert(n >=1 && n <= 31);

        Matrix32 r;

        // Build corresponding right-shift matrix.
        unsigned value = (unsigned) (1 << n);
        for (int i = 31; i >= 0; --i) {
            r.bits_[i] = value;
            value <<= 1;
        }

        return r;
    }

    bool Matrix32::operator==(const Matrix32& m) const
    {
        // Test for equality.
        for (int i = 0; i < 32; ++i)
            if (bits_[i] != m.bits_[i])
                return false;
        return true;
    }

    Matrix32 Matrix32::operator+(const Matrix32& m) const
    {
        Matrix32 r;

        // Calculate bitwise sum, in modulo-2 bit arithmetic.
        for (int i = 0; i < 32; ++i)
            r.bits_[i] = bits_[i] ^ m.bits_[i];

        return r;
    }

    Vector32 Matrix32::operator*(const Vector32& v) const
    {
        Vector32 r;

        // Calculate bitwise matrix-vector product, in modulo-2 bit
        // arithmetic.
        for (int i = 0; i < 32; ++i) 
            r.bits_ |= (Vector32(bits_[i]) * v) << (31 - i);

        return r;
    }

    Matrix32 Matrix32::operator*(const Matrix32& m) const
    {
        Matrix32 r;

        // Calculate bitwise matrix-matrix product, in modulo-2 bit
        // arithmetic.
        for (int i = 0; i < 32; ++i) {
            Vector32 c;
            for (int j = 0; j < 32; ++j)
                c.bits_ |= ((m.bits_[j] >> (31 - i)) & 0x1) << (31 - j);
            for (int j = 0; j < 32; ++j)
                r.bits_[j] |= (Vector32(bits_[j]) * c) << (31 - i);
        }

        return r;
    }

}}}}


