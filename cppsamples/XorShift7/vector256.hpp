//
// Copyright (c) 2011 QuantAlea GmbH.  All rights reserved.
// 

#ifndef MATH_F2_VECTOR256_HPP
#define MATH_F2_VECTOR256_HPP

namespace alea { namespace cuda { namespace math { namespace f2 {

    /*!
       Vector32 represents length-32 bit-vector for modulo-2 bit
       arithmetic.
     */
    struct Vector256
    {
        //! Bit-vector is represented as an length-8 array of 32-bit
        //! unsigned numbers.
        unsigned bits_[8];

        /*!
           Default class constructor, creates bit-vector with all bits
           set to 0.
         */
        Vector256(void);

        /*!
           Alternative constructor, creates bit-matrix with bit values
           set to given values.
           @param bits pointer to an array of 8 unsigned numbers
           representing bit values to initialize given bit-vector with
         */
        Vector256(const unsigned* bits);

        /*!
           Overloaded equality operator.
           @param v the bit-vector with which given bit-vector is
           compared for equality
           @return true if all bits in given bit-vector equal to bits of
           the other bit-vector, false otherwise
         */
        bool operator==(const Vector256& v) const;

        /*!
           Overloaded addition operator.
           @param v the bit-vector added, in modulo-2 bit arithmetic, to
           given bit-vector
           @return bit-vector representing the sum
         */
        Vector256 operator+(const Vector256& v) const;

        /*!
           Overloaded multiplication operator; calculates dot-product,
           in modulo-2 bit arithmetic.
           @param v the bit-vector multiplied with given bit-vector
           @return LSB bit of the returned value represents the
           dot-product, other bits of result are set to 0
         */
        unsigned operator*(const Vector256& v) const;
    };

}}}}

#endif

