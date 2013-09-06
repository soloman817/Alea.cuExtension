//
// Copyright (c) 2011 QuantAlea GmbH.  All rights reserved.
// 

#ifndef MATH_F2_MATRIX32_HPP
#define MATH_F2_MATRIX32_HPP

#include "vector32.hpp"

namespace alea { namespace cuda { namespace math { namespace f2 {

    /*!
       Matrix32 represents 32x32 bit-matrix for modulo-2 bit arithmetic.
     */
    struct Matrix32 
    {
        //! Bit-matrix is represented, in row-major order, as an
        //! lenght-32 array (rows) of 32-bit unsigned numbers (columns).
        unsigned bits_[32];

        /*!
           Default class constructor, creates bit-matrix with all bits
           set to 0.
         */
        Matrix32(void);

        /*!
           Alternative constructor, creates bit-matrix with bit values
           set to given values.
           @param bits pointer to an array of 32 unsigned numbers
           representing bit values (in row-major order) to initialize
           given bit-matrix with
         */
        Matrix32(const unsigned* bits);

        /*!
           Static function for creating identity 32x32 bit-matrix.
           @return identity bit-matrix
         */
        static Matrix32 identity(void);

        /*!
           Static function for creating bit-matrix for right-shifting a
           bit-vector (left-shifting a bit-vector for given number of
           positions would be equivalent to multiplying it by this
           matrix).
           @param n number of positions to shift (must be in [1,31]
           range)
           @return corresponding left shfit bit-matrix
         */
        static Matrix32 left(const int n);

        /*!
           Static function for creating bit-matrix for right-shifting a
           bit-vector (right-shifting a bit-vector for given number of
           positions would be equivalent to multiplying it by this
           matrix).
           @param n number of positions to shift (must be in [1,31]
           range)
           @return corresponding right shfit bit-matrix
         */
        static Matrix32 right(const int n);

        /*!
           Overloaded equality operator.
           @param m the bit-matrix with which given bit-matrix is
           compared for equality
           @return true if all bits in given bit-matrix equal to bits of
           the other bit-matrix, false otherwise
         */
        bool operator==(const Matrix32& m) const;

        /*!
           Overloaded addition operator.
           @param m the bit-matrix added, in modulo-2 bit arithmetic, to
           given bit-matrix
           @return bit-matrix representing the sum
         */
        Matrix32 operator+(const Matrix32& m) const;

        /*!
           Overloaded multiplication operator, for matrix-vector
           multiplication.
           @param v the bit-vector multiplied, in modulo-2 bit
           arithmetic, by given bit-matrix
           @return bit-vector representing matrix-vector product
         */
        Vector32 operator*(const Vector32& v) const;

        /*!
           Overloaded multiplication operator, for matrix-matrix
           multiplication.
           @param m the bit-matrix multiplied, in modulo-2 bit
           arithmetic, by given bit-matrix
           @return bit-matrix representing matrix-matrix product
         */
        Matrix32 operator*(const Matrix32& m) const;

    };

}}}}

#endif

