//
// Copyright (c) 2011 QuantAlea GmbH.  All rights reserved.
// 

#ifndef RANDOM_XORSHIFT7_JUMP_AHEAD_HPP
#define RANDOM_XORSHIFT7_JUMP_AHEAD_HPP

/*!
   The declaration for an array of unsigned 32-bit numbers, representing
   sequence of 256x256 bit-matrices (in row major order) needed for
   xorshift7 RNG jump-ahead calculations.  These matrices are calculated
   by xorshift7_jump_ahead program from f2 subdirectory of the source
   tree.  The sequence consists of xorshift7 RNG state update matrix
   raised to powers of 2 from interval [224,255], which means that, if
   xorshift7 RNG state update matrix is denoted by M, first matrix in
   sequence is M^(2^224), then M^(2^225) etc. up to M^(2^255).
 */
extern "C" const unsigned xorshift7JumpAheadMatrices[65536];

#endif
