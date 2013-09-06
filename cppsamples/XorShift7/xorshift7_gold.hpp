//
// Copyright (c) 2011 QuantAlea GmbH.  All rights reserved.
//  

#ifndef MATH_F2_XORSHIFT7_HPP
#define MATH_F2_XORSHIFT7_HPP

#include "matrix256.hpp"

namespace alea { namespace cuda { namespace math { namespace random {

    using namespace alea::cuda::math::f2;

	/*! 
		Initializes state.

		See http://www.iro.umontreal.ca/~lecuyer/myftp/papers/xorshift.pdf

		The generator state ins in x.
	*/
	void xorshift7Init (unsigned int x[8], unsigned int *init);

	/*!
		Advances by one step and returns a number in [0,1).
		
		See http://www.iro.umontreal.ca/~lecuyer/myftp/papers/xorshift.pdf

		The generator state ins in x and will be updated.
	*/
	double xorshift7Next (unsigned int x[8]);

    /*!
       Xorshift7 represents the implementation of xorshift7 RNG, as
       described in

       http://www.iro.umontreal.ca/~lecuyer/myftp/papers/xorshift.pdf.
     */
    struct Xorshift7Gold
    {
        //! The xorshift7 RNG state is represented by 8 unsigned 32-bit
        //! numbers.
        unsigned state_[8];

        //! The array of numbers used for xorshift7 state is used as
        //! circular buffer, with following member variable pointing to
        //! last (next to be reused) number in the buffer.
        int index_;

        /*!
           Default class constructor, creates RNG with all state bits
           set to 0.
         */
        Xorshift7Gold(void);

        /*!
           Alternative constructor, creates xorshift RNG with state
           initialized from sequence of random numbers, generated using
           linear congruential RNG, and starting with the given seed.
           @param seed the seed for linear congruential RNG, used to
           produce an array of 8 unsigned numbers representing initial
           state of the xorhsift7 RNG
         */        
        Xorshift7Gold(const unsigned seed);

        /*!
           Another alternative constructor, creates xorshift RNG with
           state set to values from given array of numbers (and index to
           last number set to 0).
           @param state pointer to an array of 8 unsigned numbers
           representing values to initialize RNG state with
         */        
        Xorshift7Gold(const unsigned* state);

        /*!
           The state updates of xorshift7 RNG could be calculated by
           multiplying current state (imagined as length-256 bit-vector,
           with least significant 32 bits represented by last number in
           the state buffer, next 32 bits represented by next-to-last
           number in the buffer, etc.) with specific 256x256 matrix, and
           this function is building and returning this matrix.  As
           given matrix is rather sparse, state is updated much faster
           through direct calculations, implemented by
           Xorshift7::getUniform() method; however, this state update
           matrix is usable for jumping ahead RNG for large number of
           steps.
           @return xorshift7 RNG state update matrix
         */
        static Matrix256 getMatrix(void);
        
        /*!
           Copy numbers representing xorshift7 RNG state to given array.
           @param state array of 8 unsigned 32-bit numbers to copy RNG
           state to
         */
        void getState(unsigned* state);

        /*!
           Produce next random number from uniform distribution, as an
           unsigned number from the range of all 32-bit unsigned
           numbers, and update RNG state.
           @return the random number generated
         */
        unsigned getUnsignedUniform(void);

        /*!
           Produce next random number from uniform distribution, as a
           real number from [0,1] interval, and update RNG state.
           @return the random number generated 
         */
        float getFloatUniform(void);

        /*!
           Produce next two random numbers from normal distribution with
           mean equal to 0 and standard deviation equal to 1, and update
           RNG state accordingly
           @param value0 pointer to store first random number generated
           @param value1 pointer to store second random number generated
         */
        void getFloatNormal(float* value0, float* value1);
    };

}}}}

#endif

