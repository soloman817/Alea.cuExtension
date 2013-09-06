//#include "xorshift7.hpp"

//
// Copyright (c) 2011 QuantAlea GmbH.  All rights reserved.
// 

// The RNG is producing random numbers uniformly over the range of
// unsigned data type; in order to have real numbers in [0,1] range
// produced, these are multiplied by following value.
#define SCALING_FACTOR (1.0f / 4294967295.0f)

// The pi value, used for producing random numbers from normal
// distribution.
#define PI 3.14159265358979f

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
    __device__ __host__ int xorshift7GetSize(void)
    {
        return sizeof(Xorshift7);
    }
    
    /*!
     * Cast given pointer to the pointer to xorshift7 RNG data
     * structure.
     * \param buffer pointer to the segment of memory allocated to be
     * used as xorshift7 RNG data structure
     * \return above pointer casted to the pointer to xorshift7 RNG data
     * structure
     */
    __device__ Xorshift7* xorshift7Bless(void* buffer)
    {
        return (Xorshift7*) buffer;
    }

    /*!
     * Initialize xorshift7 RNG state to given values.
     * \param xorshift7 pointer to the RNG data structure
     * \param buffer an array of 8 unsigned 32-bit numbers to initialize
     * RNG state with
     */
    __device__ void xorshift7Init(Xorshift7* xorshift7, const unsigned* buffer)
    {
        // Copy given values into the RNG state.
        for (int i = 0; i < 8; ++i)
            xorshift7->state_[i] = buffer[i];
        
        // Initialize index of the "oldest" number in the RNG state.
        xorshift7->index_ = 0;
    }

    /*!
     * Generate random number from uniform distribution, and update RNG
     * state accordingly.
     * \param xorshift7 pointer to the RNG data structure
     * \return random number, uniformly distributed over [0,1] range
     */
    __device__ float xorshift7GetUniform(Xorshift7* xorshift7)
    {
        unsigned r;
        
        // Calculate next random number, and update xorshift7 RNG state.
        // This code is directly following xorshift7 RNG definition, as
        // described in
        // http://www.iro.umontreal.ca/~lecuyer/myftp/papers/xorshift.pdf
        // paper.
        unsigned t;
        t = xorshift7->state_[(xorshift7->index_ + 7) & 0x7];
        t = t ^ (t << 13);
        r = t ^ (t << 9);
        t = xorshift7->state_[(xorshift7->index_ + 4) & 0x7];
        r ^= t ^ (t << 7);
        t = xorshift7->state_[(xorshift7->index_ + 3) & 0x7];
        r ^= t ^ (t >> 3);
        t = xorshift7->state_[(xorshift7->index_ + 1) & 0x7];
        r ^= t ^ (t >> 10);
        t = xorshift7->state_[xorshift7->index_];
        t = t ^ (t >> 7);
        r ^= t ^ (t << 24);
        xorshift7->state_[xorshift7->index_] = r;
        xorshift7->index_ = (xorshift7->index_ + 1) & 0x7;
        
        // Convert calculated unsigned random number to the real number
        // from [0,1] interval.
        return (float) r * SCALING_FACTOR;
    }
    
    __device__ unsigned xorshift7GetUniformUnsigned(Xorshift7* xorshift7)
    {
        unsigned r;
        
        // Calculate next random number, and update xorshift7 RNG state.
        // This code is directly following xorshift7 RNG definition, as
        // described in
        // http://www.iro.umontreal.ca/~lecuyer/myftp/papers/xorshift.pdf
        // paper.
        unsigned t;
        t = xorshift7->state_[(xorshift7->index_ + 7) & 0x7];
        t = t ^ (t << 13);
        r = t ^ (t << 9);
        t = xorshift7->state_[(xorshift7->index_ + 4) & 0x7];
        r ^= t ^ (t << 7);
        t = xorshift7->state_[(xorshift7->index_ + 3) & 0x7];
        r ^= t ^ (t >> 3);
        t = xorshift7->state_[(xorshift7->index_ + 1) & 0x7];
        r ^= t ^ (t >> 10);
        t = xorshift7->state_[xorshift7->index_];
        t = t ^ (t >> 7);
        r ^= t ^ (t << 24);
        xorshift7->state_[xorshift7->index_] = r;
        xorshift7->index_ = (xorshift7->index_ + 1) & 0x7;
        
        // Convert calculated unsigned random number to the real number
        // from [0,1] interval.
        return r;
    }

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
    __device__ void xorshift7GetNormal(Xorshift7* xorshift7, float* value0, float* value1)
    {
        // Generate two uniformly distributed random numbers, and apply
        // Box-Muller transformation to convert these numbers to
        // normally distributed random numbers.
        float r = sqrtf(-2 * logf(xorshift7GetUniform(xorshift7)));
        float phi = 2 * PI * xorshift7GetUniform(xorshift7);
        *value0 = r * cosf(phi);
        *value1 = r * sinf(phi);
    }

}}}}
