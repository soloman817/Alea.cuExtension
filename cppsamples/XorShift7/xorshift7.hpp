#ifndef XORSHIFT7_HPP
#define XORSHIFT7_HPP

//
// Copyright (c) 2011 QuantAlea GmbH.  All rights reserved.
// 

namespace alea { namespace cuda { namespace math { namespace random 
{


    /*!
     * The data structure representing xorshift7 RNG.  The RNG is
     * described by its current state, that in the case of xorshift7 RNG
     * practially consists of last 8 random numbers generated.
     */
    struct Xorshift7
    {
        //! The RNG state; this array is used as circular buffer.
        unsigned state_[8];
        //! The index of the "oldest" (next to be reused) number in the
        //! above array.
        int index_;
    };

    /*!
     * Return the size of the data structure representing the RNG.
     * \return the size of the xorshift7 RNG data structure
     */
    __device__ __host__ int xorshift7GetSize(void);
    
    /*!
     * Cast given pointer to the pointer to xorshift7 RNG data
     * structure.
     * \param buffer pointer to the segment of memory allocated to be
     * used as xorshift7 RNG data structure
     * \return above pointer casted to the pointer to xorshift7 RNG data
     * structure
     */
    __device__ Xorshift7* xorshift7Bless(void* buffer);

    /*!
     * Initialize xorshift7 RNG state to given values.
     * \param xorshift7 pointer to the RNG data structure
     * \param buffer an array of 8 unsigned 32-bit numbers to initialize
     * RNG state with
     */
    __device__ void xorshift7Init(Xorshift7* xorshift7, const unsigned* buffer);

    /*!
     * Generate random number from uniform distribution, and update RNG
     * state accordingly.
     * \param xorshift7 pointer to the RNG data structure
     * \return random number, uniformly distributed over [0,1] range
     */
    __device__ float xorshift7GetUniform(Xorshift7* xorshift7);
    
    /*!
     * Generate 2 random numbers from normal distribution, with mean
     * equal 0 and standard deviation equal 1.  Box-Muller
     * transformation is used to produce normally distributed random
     * numbers from pair of uniformly distributed random numbers
     * initially generated.  Because it is not possible to have static
     * variables in device functions (that would be employed for saving
     * one of these numbers for later), this function is return both
     * numbers immediately, thus pushing burden of managing them to the
     * callee.
     * \param xorshift7 pointer to the RNG data structure
     * \param value0 pointer to the location to store first random
     * number generated
     * \param value1 pointer to the location to store second random
     * number generated
     */
    __device__ void xorshift7GetNormal(Xorshift7* xorshift7, float* value0, float* value1);

}}}}

#endif