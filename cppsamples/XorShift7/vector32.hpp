//
// Copyright (c) 2011 QuantAlea GmbH.  All rights reserved.
//  

#ifndef MATH_F2_VECTOR32_HPP
#define MATH_F2_VECTOR32_HPP

namespace alea { namespace cuda { namespace math { namespace f2 {

    /*!
       Vector32 represents length-32 bit-vector for modulo-2 bit
       arithmetic.
     */
    struct Vector32
    {
        //! Bit-vector is represented as an 32-bit unsigned number.
        unsigned bits_;
        
        /*!
           Default class constructor, creates bit-vector with all bits
           set to 0.
         */
        Vector32(void);

        /*!
           Alternative constructor, creates bit-vector with bit values
           set to given values.
           @param bits 32-bit unsigned number representing bit values to
           initialize given bit-vector with
         */
        Vector32(const unsigned bits);

        /*!
           Overloaded equality operator.
           @param v the bit-vector with which given bit-vector is
           compared for equality
           @return true if all bits in given bit-vector equal to bits of
           the other bit-vector, false otherwise
         */
        bool operator==(const Vector32& v) const;
        
        /*!
           Overloaded addition operator.
           @param v the bit-vector added, in modulo-2 bit arithmetic, to
           given bit-vector
           @return bit-vector representing the sum
         */
        Vector32 operator+(const Vector32& v) const;

        /*!
           Overloaded multiplication operator; calculates dot-product,
           in modulo-2 bit arithmetic.
           @param v the bit-vector multiplied with given bit-vector
           @return LSB bit of the returned value represents the
           dot-product, other bits of result are set to 0
         */
        unsigned operator*(const Vector32& v) const;
    };

}}}}

#endif

