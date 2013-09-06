//
// Copyright (c) 2011 QuantAlea GmbH.  All rights reserved.
// 

#include <algorithm>
#include <cassert>
#include <iostream> 
#include <iomanip>

#include "matrix256.hpp"

namespace alea { namespace cuda { namespace math { namespace f2 {

    Matrix256::Matrix256(void)
    {
        // Initialize all bits to 0.
        for (int i = 0; i < 256; ++i)
            std::fill(bits_[i], bits_[i] + 8, 0);
    }

    Matrix256::Matrix256(const unsigned* bits)
    {
        // Initialize all bits to given values.
        for (int i = 0; i < 256; ++i)
            std::copy(bits + 8 * i, bits + 8 * (i + 1), bits_[i]);
    }

    Matrix256 Matrix256::identity(void)
    {
        Matrix256 r;

        // Build identity matrix.
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 32; ++j)
                for (int k = 0; k < 8; ++k)
                    r.bits_[i * 32 + j][k] = (i == k) ? 1 << (31 - j) : 0;

        return r;
    }

	void printMat(const Matrix32 & m)
	{
		for (int i = 0; i < 32; ++i) {
			std::cout << std::hex << std::setw(8) << m.bits_[i]<< std::endl;
		}
	}

    void Matrix256::set32x32Block(const int row, const int col,
                                  const Matrix32& b)
    {
		//std::cout << "set32x32Block row = " << row << ", col = " << col << std::endl;
		//printMat(b);

        // Verify that function arguments are valid.
        assert(row >=0 && row < 8);
        assert(col >=0 && col < 8);

        // Initialize this particular block of bits within given matrix
        // to the specified values.
        for (int i = 0; i < 32; ++i)
            bits_[32 * row + i][col] = b.bits_[i];
    }

    bool Matrix256::operator==(const Matrix256& m) const
    {
        // Test for equality.
        for (int i = 0; i < 256; ++i)
            for (int j = 0; j < 8; ++j)
                if (bits_[i][j] != m.bits_[i][j])
                    return false;
        return true;
    }
    
    Matrix256 Matrix256::operator+(const Matrix256& m) const
    {
        Matrix256 r;

        // Calculate bitwise sum, in modulo-2 bit arithmetic.
        for (int i = 0; i < 256; ++i)
            for (int j = 0; j < 8; ++j)
                r.bits_[i][j] = bits_[i][j] ^ m.bits_[i][j];

        return r;
    }

    Vector256 Matrix256::operator*(const Vector256& v) const
    {
        Vector256 r;

        // Calculate bitwise matrix-vector product, in modulo-2 bit
        // arithmetic.
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 32; ++j)
			{
				Vector256 t(bits_[i*32 + j]);
                r.bits_[i] |= (t * v) << (31 - j);
			}

        return r;
    }

    Matrix256 Matrix256::operator*(const Matrix256& m) const
    {
        Matrix256 r;

        // Calculate bitwise matrix-matrix product, in modulo-2 bit
        // arithmetic.
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 32; ++j) {
                // Calculate bit values of the bit-vector representing
                // current column of the other matrix.
                unsigned bits[8];
                std::fill(bits, bits + 8, 0);
                for (int k = 0; k < 8; ++k)
                    for (int l = 0; l < 32; ++l)
                        bits[k] |= ((m.bits_[k * 32 + l][i] >> (31 - j)) & 0x1) << (31 - l);

                // Bits of the current column of the resulting matrix
                // are calculated as vector products of the bit-vector
                // representing current row of given matrix and the
                // bit-vector representing current column of the other
                // matrix.
                Vector256 c(bits);
                for (int k = 0; k < 256; ++k)
                    r.bits_[k][i] |= (Vector256(bits_[k]) * c) << (31 - j);
            }

        return r;
    }

    Matrix256 Matrix256::powPow2(const int p) const
    {
        // Verify that function arguments is valid.
        assert(p >= 0);

        Matrix256 r = *this;

        // Raise given matrix to the power of 2 specified.  
        for (int i = 0; i < p; ++i)
            r = r * r;

        return r;
    }
    
    Matrix256 Matrix256::pow(const int n) const
    {
        // Verify that function arguments is valid.
        assert(n >= 0);

        Matrix256 r = Matrix256::identity();

        // Raise given matrix to the number specified.  Standard
        // square-and-multiply exponentiation technique is used.
        Matrix256 m = *this;
        for (int k = n; k != 0; k >>= 1) {
            if (k & 0x1)
                r = r * m;
            m = m * m;
        }

        return r;
    }

}}}}

