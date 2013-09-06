//
// Copyright (c) 2011 QuantAlea GmbH.  All rights reserved.
// 

#ifndef MATH_F2_MATRIX256_HPP
#define MATH_F2_MATRIX256_HPP

#include "matrix32.hpp"
#include "vector256.hpp"

namespace alea { namespace cuda { namespace math { namespace f2 {

    /*!
       Matrix256 represents 256x256 bit-matrix for modulo-2 bit
       arithmetic.
     */
    struct Matrix256 
    {
        //! Bit-matrix is represented, in row-major order, as an
        //! lenght-256 array (rows) of 8 32-bit unsigned numbers
        //! (columns).
        unsigned bits_[256][8];

        /*!
           Default class constructor, creates bit-matrix with all bits
           set to 0.
         */
        Matrix256(void);

        /*!
           Alternative constructor, creates bit-matrix with bit values
           set to given values.
           @param bits pointer to an array of 2048 unsigned numbers
           representing bit values (in row-major order) to initialize
           given bit-matrix with
         */
        Matrix256(const unsigned* bits);

        /*!
           Static function for creating identity 256x256 bit-matrix.
           @return identity bit-matrix
         */
        static Matrix256 identity(void);

        /*!
           The 256x256 bit-matrix consists of 8x8 grid of 32x32 blocks
           of bits; this function is changing bit-values of specific
           block in given matrix.
           @param row the index of the block row (must be in [0,7] range
           @param col the index of the block column (must be in [0,7]
           range
           @param b 32x32 bit matrix, containing bit values to initalize
           this particular block of given matrix with.
         */
        void set32x32Block(const int row, const int col, const Matrix32& b);

        /*!
           Overloaded equality operator.
           @param m the bit-matrix with which given bit-matrix is
           compared for equality
           @return true if all bits in given bit-matrix equal to bits of
           the other bit-matrix, false otherwise
         */
        bool operator==(const Matrix256& m) const;

        /*!
           Overloaded addition operator.
           @param m the bit-matrix added, in modulo-2 bit arithmetic, to
           given bit-matrix
           @return bit-matrix representing the sum
         */
        Matrix256 operator+(const Matrix256& m) const;

        /*!
           Overloaded multiplication operator, for matrix-vector
           multiplication.
           @param v the bit-vector multiplied, in modulo-2 bit
           arithmetic, by given bit-matrix
           @return bit-vector representing matrix-vector product
         */
        Vector256 operator*(const Vector256& v) const;

        /*!
           Overloaded multiplication operator, for matrix-matrix
           multiplication.
           @param m the bit-matrix multiplied, in modulo-2 bit
           arithmetic, by given bit-matrix
           @return bit-matrix representing matrix-matrix product
         */
        Matrix256 operator*(const Matrix256& m) const;

        /*!
           Raise given matrix to the specific power of 2.
           @param p power of 2 to raise given matrix to
           @return resulting matrix
         */
        Matrix256 powPow2(const int p) const;
        
        /*!
           Raise given matrix to the specific number
           @param n number to raise given matrix to
           @return resulting matrix
         */
        Matrix256 pow(const int n) const;
    };

}}}}

#endif

