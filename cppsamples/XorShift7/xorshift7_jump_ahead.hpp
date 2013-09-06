//
// Copyright (c) 2011 QuantAlea GmbH.  All rights reserved.
//  
#ifndef XORSHIFT7_JUMP_AHEAD_HPP
#define XORSHIFT7_JUMP_AHEAD_HPP

#include <cassert>
#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>

#include "xorshift7_gold.hpp"
#include "matrix256.hpp"

namespace alea { namespace cuda { namespace math { namespace random {

	// This function is dumping xorshift7 RNG state matrix, raised to
	// specific range of powers of 2, to standard output; these matrices are
	// useful for xorshift7 RNG jump-ahead calculations.  Given matrices are
	// dumped as a large array of unsigned numbers, in row-major order.  The
	// program is expecting 2 arguments from the command line: these are
	// boundaries of interval of powers of 2 that xorshift7 RNG state matrix
	// will be raised to, and then dumped to standard output.  Lower
	// boundary of this interval must be non-negative, and upper boundary
	// must be greater of equal to the lower bound.

	void xorshift7JumpHeaderGenerateC(int lo, int hi);
	void xorshift7JumpHeaderGenerateFSharp(int lo, int hi);

}}}}

#endif