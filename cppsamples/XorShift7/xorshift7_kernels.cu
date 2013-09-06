//
// Copyright (c) 2011 QuantAlea GmbH.  All rights reserved.
// 

#include "xorshift7.cuh"
#include "xorshift7_jump_ahead_matrices.hpp"
#include "xorshift7_kernels.hpp"
#include <time.h>
#include <stdio.h>

// The parameters of the linear congruential RNG, used to initialize state of the xorshift7 RNG
// these parametrs are as from "Numerical Recipes" book 
// see http://en.wikipedia.org/wiki/Linear_congruential_generator
#define LCG_A 1664525U
#define LCG_C 1013904223U

namespace alea { namespace cuda { namespace math { namespace random 
{
    /*!
        The function for jumping ahead xorshift7 RNG state from the
        overall start state to the start state corresponding to the
        thread index within given number of all threads that will run
        xorshift7 RNG afterwards.
        \param numThreads the number of all threads that will run
        xorshift7 RNG
        \param threadRank the index of the thread for which xoshift7 RNG
        start state is calculated
        \param stateStart the overall start state of the xorshift7 RNG
        \param jumpAheadMatrices pointer to an array, presumably stored
        in device global memory, of unsigned 32-bit numbers, representing
        sequence of pre-calculated 256x256 bit-matrices (stored in
        row-major order), needed for xorshift7 RNG jump-ahead
        calculations
        \param state pointer to memory where calculated xorshift7 RNG
        start state for given thread will be stored
        \param jumpAheadMatrixCache pointer to 256-bit block, in shared
        memory, to be used as cache for current row of the jump ahead
        matrix
     */
     __device__ void xorshift7JumpAhead(
         int            numThreads
       , int            threadRank
       , const unsigned * stateStart
       , const unsigned * jumpAheadMatrices
       , unsigned       * jumpAheadMatrixCache
       , unsigned       * state
       )
    {
        // Calculate the number of threads per block and the rank of
        // thread within a block.
        int numThreadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
        int threadRankInBlock = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;

        // Calculate first power of 2 greater or equal to the number of
        // threads; if that number is p, than each thread should start
        // with the state that is 2^(256-p) steps distant from states of
        // its neighboring (by rank) threads.  Jumping ahead to the
        // corresponding state could be accomplished by multiplying
        // initial RNG state, interpreted as lenght-256 bit-vector, by
        // corresponding update matrix.  If xorshift7 RNG specific
        // 256x256 state update bit-matrix is denoted by M, and initial
        // state bit-vector by v, then initial state of the thread with
        // rank 0 could be calculated as (M^(2^(256-p)))^0*v, initial
        // state of thread with rank 1 as (M^(2^(256-p)))^1*v, initial
        // state of thread with rank 2 as (M^(2^(256-p)))^2*v, etc.
        // Thus, matrix M^(2^(256-p)) appears as base matrix for these
        // calculations. Matrices M^(2^224), M^(2^225) etc. up to
        // M^(2^255) are pre-calculated and, having in mind that number
        // of threads is integer number, thus value of p is less or
        // equal to 31, matrix M^(2^(256-p)) (let's denote it B) is
        // among these, so the pointer to this matrix is initialized
        // here to point to corresponding matrix in the sequence of
        // pre-calculated matrices supplied as argument to the kernel.
        int p = 0;
        for ( ; (1 << p) < numThreads; ++p)
            ;
        int matrixSize = 256 * 8;
        const unsigned* matrix = jumpAheadMatrices + (32 - p) * matrixSize;
        
        // Initialize RNG state to start state.
        for (int i = 0; i < 8; ++i)
            state[i] = stateStart[i];
        
         //Jump ahead, according to the thread rank, to RNG state
         //appopriate for given thread; if thread rank is denoted by r,
         //then jumping ahead could be accomplished through multiplying
         //start state by matrix B raised to the thread rank: B^r*v.
         //Thread rank could be interpreted as base-2 number
         //b[p-1]...b[1]b[0] (where b[0] is the least significant bit,
         //etc.), and then state update could be interpreted as follows:
         //  B^r*v=B^(b[p-1]*2^(p-1)+...+b[1]*2^1+b[0]*2^0)*v=
         //       =B^(b[p-1]*2^(p-1))*...*B^(b[1]*2^1)*B^(b[0]*2^0)*v
         //       =(B^(2^(p-1)))^b[p-1]*...*(B^(2^1)^b[1]*(B^(2^0))^b[0]*v
         //All of matrices (B^*(2^0)), (B^(2^1)), ..., B^(2^(p-1)) are
         //pre-calculated, so the above expression could be calculated in
         //a loop, from right-to-left.
        for (int i = 0; i < p; ++i) {
            // Current state has to be used as previous state for the
            // next current state.
            unsigned statePrev[8];
            for (int j = 0; j < 8; ++j)
                statePrev[j] = state[j];
                
            // Calculate product of current bit-matrix with bit-vector
            // representing previous state.  Resulting bits will be
            // calculated in 8 groups of 32 bits.
            for (int j = 0; j < 8; ++j) {
                // Initialize current group of bits of the result
                // vector.
                unsigned stateWord = 0;

                // Each bit of result is calculated as vector product of
                // corresponding row-vector of the current matrix, and
                // the previous state bit-vector.
                for (int k = 0; k < 32; ++k) {
                    // Read the next row of the jump-ahead matrix into
                    // the shared memory cache.  The first 8 thread of
                    // the block are involved into reading the matrix
                    // row elements; as the warp size should be greater
                    // than 8, there should be always enough threads to
                    // do the work.
                    __syncthreads();
                    for (int l = threadRankInBlock; l < 8; l += numThreadsPerBlock)
                        jumpAheadMatrixCache[l] = matrix[l];
                    matrix += 8;

                    // Continue only if current bit of the thread index
                    // is 1.
                    __syncthreads();
                    if (threadRank & (1 << i) == 0)
                        continue;

                    // Bit-vectors product will be calculated as sum of
                    // products of individual bits.  For efficiency,
                    // this calculation will be accomplished through
                    // calculating partial sums first, where each
                    // partial sums will accumulate sums of products of
                    // 8 pairs of bit, where these pairs are 32
                    // positions apart from each other.
                    unsigned partialSums = 0;
                    for (int l = 0; l < 8; ++l)
                        partialSums ^= jumpAheadMatrixCache[l] & statePrev[l];
                        
                    // Now, calculate single-bit total sum from above
                    // partial sums.
                    unsigned sum = partialSums;
                    sum = (sum >> 16) ^ (sum & 0xffff);
                    sum = (sum >> 8) ^ (sum & 0xff);
                    sum = (sum >> 4) ^ (sum & 0xf);
                    sum = (sum >> 2) ^ (sum & 0x3);
                    sum = (sum >> 1) ^ (sum & 0x1);
                        
                    // Update current group of bits of the result vector
                    // with the vector product calculated
                    // above.
                    stateWord <<= 1;
                    stateWord |= sum;
                }

                // The current state vector is updated only if current
                // bit of the thread index is 1.
                if (threadRank & (1 << i))
                    // Copy over current group of bits of the result
                    // vector into the current state.
                    state[j] = stateWord;
            }
        }
    }

    /*!
        Kernel for generating uniformly distributed random numbers using
        xorshift7 RNG.  The kernel is ready for execution on multiple
        devices.  It will initialize RNG with the proper state, according
        to the thread rank within all threads run on all devices, and
        then will generate given number of random values, writing them
        into the appropriate locations of device global memory.  The
        order of write is per device interleaved, which means that first
        thread on given device will write the first number generated into
        the first location in the array pre-allocated for results, the
        second thread will write its first number generated on the second
        location, etc.
        \param numRuns the number of times that this kernel will be
        re-run
        \param runRank the index of the run on which this particular
        instance of kernel is run
        \param stateStart pointer to an array, presumably stored in
        device global memory, of 8 unsigned 32-numbers, representing the
        start state for the xorshift7 RNG
        \param jumpAheadMatrices pointer to an array, presumably stored
        in device global memory, of unsigned 32-bit numbers, representing
        sequence of pre-calculated 256x256 bit-matrices (stored in
        row-major order), needed for xorshift7 RNG jump-ahead
        calculations
        \param numSteps number of random values to be generated during
        the kernel execution
        \param results pointer to an array, presumably allocated in
        device global memory, where each thread will store its random
        values generated
     */
    extern "C" __global__ void xorshift7GenerateUniformKernel(
        int                 numRuns
      , int                 runRank
      , const unsigned    * stateStart
      , const unsigned    * jumpAheadMatrices
      , const int           numSteps
      , float             * results
      )
    {
        // Shared memory declaration; aligned to 4 because of the
        // intended usage as the cache for current row of the jump-ahead
        // matrix.
        extern  __shared__  __align__(4) char sharedData[];
        
        // Calculate ranks of the block within the grid, and the thread
        // within the block, as well as rank of the thread within the
        // whole grid.
        int numBlocks = gridDim.x * gridDim.y * gridDim.z;
        int blockRank = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
        int numThreadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
        int numThreads = numBlocks * numThreadsPerBlock;
        int threadRankInBlock = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
        int threadRank = blockRank * numThreadsPerBlock + threadRankInBlock;

        // Perform jumping ahead, taking into account the total number
        // of thread on all devices, as well as current thread rank
        // within all thread on all devices.
        unsigned state[8];
        xorshift7JumpAhead(numRuns * numThreads, runRank * numThreads + threadRank, stateStart, jumpAheadMatrices, (unsigned*) sharedData, state);
        
        // Use corresponding piece of shared memory for xorshift7 RNG
        // data structure, and intialized RNG with the state calculated
        // through above jump-ahead procedure.
        Xorshift7* rng = xorshift7Bless(sharedData + threadRankInBlock * xorshift7GetSize());
        xorshift7Init(rng, state);
        
        // Generate sequence of uniformly distributed random numbers,
        // and write numbers from the sequence, in a per device
        // interleaved order, to the corresponding locations into the
        // global memory.
        int index = threadRank;
        for (int i = 0; i < numSteps; ++i) {
            results[index] = xorshift7GetUniform(rng);
            index += numThreads;
        }
    }

    extern "C" __global__ void xorshift7GenerateUniformKernelUnsigned(
        int                 numRuns
      , int                 runRank
      , const unsigned    * stateStart
      , const unsigned    * jumpAheadMatrices
      , const int           numSteps
      , unsigned          * results
      )
    {
        // Shared memory declaration; aligned to 4 because of the
        // intended usage as the cache for current row of the jump-ahead
        // matrix.
        extern  __shared__  __align__(4) char sharedData[];
        
        // Calculate ranks of the block within the grid, and the thread
        // within the block, as well as rank of the thread within the
        // whole grid.
        int numBlocks = gridDim.x * gridDim.y * gridDim.z;
        int blockRank = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
        int numThreadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
        int numThreads = numBlocks * numThreadsPerBlock;
        int threadRankInBlock = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
        int threadRank = blockRank * numThreadsPerBlock + threadRankInBlock;

        // Perform jumping ahead, taking into account the total number
        // of thread on all devices, as well as current thread rank
        // within all thread on all devices.
        unsigned state[8];
        xorshift7JumpAhead(numRuns * numThreads, runRank * numThreads + threadRank, stateStart, jumpAheadMatrices, (unsigned*) sharedData, state);
        
        // Use corresponding piece of shared memory for xorshift7 RNG
        // data structure, and intialized RNG with the state calculated
        // through above jump-ahead procedure.
        Xorshift7* rng = xorshift7Bless(sharedData + threadRankInBlock * xorshift7GetSize());
        xorshift7Init(rng, state);
        
        // Generate sequence of uniformly distributed random numbers,
        // and write numbers from the sequence, in a per device
        // interleaved order, to the corresponding locations into the
        // global memory.
        int index = threadRank;
        for (int i = 0; i < numSteps; ++i) {
            results[index] = xorshift7GetUniformUnsigned(rng);
            index += numThreads;
        }
    }

	/*!
        Kernel for generating normally distributed random numbers using
        xorshift7 RNG.  The kernel is ready for execution on multiple
        devices.  It will initialize RNG with the proper state, according
        to the thread rank within all threads run on all devices, and
        then will generate given number of random values, writing them
        into the appropriate locations of device global memory.  The
        order of write is per device interleaved, which means that first
        thread on given device will write the first number generated into
        the first location in the array pre-allocated for results, the
        second thread will write its first number generated on the second
        location, etc.
        \param numRuns the number of times that this kernel will be
        re-run
        \param runRank the index of the run on which this particular
        instance of kernel is run
        \param stateStart pointer to an array, presumably stored in
        device global memory, of 8 unsigned 32-numbers, representing the
        start state for the xorshift7 RNG
        \param jumpAheadMatrices pointer to an array, presumably stored
        in device global memory, of unsigned 32-bit numbers, representing
        sequence of pre-calculated 256x256 bit-matrices (stored in
        row-major order), needed for xorshift7 RNG jump-ahead
        calculations
        \param numSteps number of random values to be generated during
        the kernel execution
        \param results pointer to an array, presumably allocated in
        device global memory, where each thread will store its random
        values generated
     */
    extern "C" __global__ void xorshift7GenerateNormalKernel(
        int               numRuns
      , int               runRank
      , unsigned        * stateStart
      , const unsigned  * jumpAheadMatrices
      , const int         numSteps
      , float           * results
      )
    {
        // Shared memory declaration; aligned to 4 because of the
        // intended usage as the cache for current row of the jump-ahead
        // matrix.
        extern  __shared__  __align__(4) char sharedData[];
        
        // Calculate ranks of the block within the grid, and the thread
        // within the block, as well as rank of the thread within the
        // whole grid.
        int numBlocks = gridDim.x * gridDim.y * gridDim.z;
        int blockRank = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * gridDim.x + blockIdx.x;
        int numThreadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
        int numThreads = numBlocks * numThreadsPerBlock;
        int threadRankInBlock = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
        int threadRank = blockRank * numThreadsPerBlock + threadRankInBlock;

        // Perform jumping ahead, taking into account the total number
        // of thread on all devices, as well as current thread rank
        // within all thread on all devices.
        unsigned state[8];
        xorshift7JumpAhead(numRuns * numThreads, runRank * numThreads + threadRank, stateStart, jumpAheadMatrices, (unsigned*) sharedData, state);
        
        // Use corresponding piece of shared memory for xorshift7 RNG
        // data structure, and intialized RNG with the state calculated
        // through above jump-ahead procedure.
        Xorshift7* rng = xorshift7Bless(sharedData + threadRankInBlock * xorshift7GetSize());
        xorshift7Init(rng, state);
        
        // Generate sequence of normally distributed random numbers, and
        // write numbers from the sequence, in a per device interleaved
        // order, to the corresponding locations into the global memory.
        float value0, value1;
        int index = threadRank;
        int count = numSteps / 2;
        for (int i = 0; i < count; i++) {
            xorshift7GetNormal(rng, &value0, &value1);
            results[index] = value0;
            index += numThreads;
            results[index] = value1;
            index += numThreads;
        }
        if (numSteps & 0x1) {
            xorshift7GetNormal(rng, &value0, &value1);
            results[index] = value0;
        }
    }

    void rngGetStartState(
        unsigned    seed
      , unsigned  * state
      )
    {
        // Initialize RNG state using sequence of numbers generated by
        // linear congruential RNG with given seed, and set last state
        // number index to 0.
        state[0] = seed;
        for (int i = 1; i < 8; ++i)
            state[i] = LCG_A * state[i - 1] + LCG_C;
    }

    cudaError_t rngGenerateUniformKernelExecute(
        int         numRuns
      , int         runRank
      , dim3        gridSize
      , dim3        blockSize
      , unsigned    seed
      , int         numSteps
      , float     * results
      )
    {
        cudaError_t error;

        // Generate RNG initial state, and copy it to the device memory.
        unsigned stateStart[8];
        rngGetStartState(seed, stateStart);
        unsigned* stateStartDevice;
        error = cudaMalloc((void**) &stateStartDevice, 8 * sizeof(unsigned));
        if (error != cudaSuccess)
            return error;
        error = cudaMemcpy(stateStartDevice, stateStart, sizeof(stateStart), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
            return error;

        // Copy pre-calculated bit-matrices, needed for jump-ahead
        // calculations, to the device memory.
        unsigned* jumpAheadMatricesDevice;
        error = cudaMalloc((void**) &jumpAheadMatricesDevice, sizeof(xorshift7JumpAheadMatrices));
        if (error != cudaSuccess)
            return error;
        error = cudaMemcpy(jumpAheadMatricesDevice, xorshift7JumpAheadMatrices, sizeof(xorshift7JumpAheadMatrices), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
            return error;

        // Calculate shared memory size needed, and then execute the kernel.
        // Besides using the shared memory for the RNG data structure, it
        // will be used as cache for jump-ahead matrix in the corresponding
        // device function above.  Thus, the shared memory size should be at
        // least the same as the jump-ahead matrix row size; however, as the
        // size of this row is the same as the RNG state size, and as the
        // block must contain at least one thread, this requirement will be
        // always satisfied.
        int sharedMemorySize = blockSize.x * blockSize.y * blockSize.z * xorshift7GetSize();
        xorshift7GenerateUniformKernel<<<gridSize, blockSize, sharedMemorySize>>>(numRuns, runRank, stateStartDevice, jumpAheadMatricesDevice, numSteps, results);
        error = cudaGetLastError();
        if (error != cudaSuccess)
            return error;

        // Free device memory used.
        error = cudaFree(jumpAheadMatricesDevice);
        if (error != cudaSuccess)
            return error;
        error = cudaFree(stateStartDevice);
        if (error != cudaSuccess)
            return error;
        
        return error;
    }

    cudaError_t rngGenerateUniformKernelExecuteUnsigned(
        int         numRuns
      , int         runRank
      , dim3        gridSize
      , dim3        blockSize
      , unsigned    seed
      , int         numSteps
      , unsigned  * results
      )
    {
        cudaError_t error;

        // Generate RNG initial state, and copy it to the device memory.
        unsigned stateStart[8];
        rngGetStartState(seed, stateStart);
        unsigned* stateStartDevice;
        error = cudaMalloc((void**) &stateStartDevice, 8 * sizeof(unsigned));
        if (error != cudaSuccess)
            return error;
        error = cudaMemcpy(stateStartDevice, stateStart, sizeof(stateStart), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
            return error;

        // Copy pre-calculated bit-matrices, needed for jump-ahead
        // calculations, to the device memory.
        unsigned* jumpAheadMatricesDevice;
        error = cudaMalloc((void**) &jumpAheadMatricesDevice, sizeof(xorshift7JumpAheadMatrices));
        if (error != cudaSuccess)
            return error;
        error = cudaMemcpy(jumpAheadMatricesDevice, xorshift7JumpAheadMatrices, sizeof(xorshift7JumpAheadMatrices), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
            return error;

        // Calculate shared memory size needed, and then execute the kernel.
        // Besides using the shared memory for the RNG data structure, it
        // will be used as cache for jump-ahead matrix in the corresponding
        // device function above.  Thus, the shared memory size should be at
        // least the same as the jump-ahead matrix row size; however, as the
        // size of this row is the same as the RNG state size, and as the
        // block must contain at least one thread, this requirement will be
        // always satisfied.
        int sharedMemorySize = blockSize.x * blockSize.y * blockSize.z * xorshift7GetSize();
		clock_t t0 = clock();
		printf("\n\n");
		printf("gridSize = (%d,%d,%d)\n", gridSize.x, gridSize.y, gridSize.z);
		printf("blockSize = (%d,%d,%d)\n", blockSize.x, blockSize.y, blockSize.z);
		printf("sharedSize = %d\n", sharedMemorySize);
		printf("numRuns = %d\n", numRuns);
		printf("runRank = %d\n", runRank);
		printf("numSteps = %d\n", numSteps);
		printf("\n\n");
        xorshift7GenerateUniformKernelUnsigned<<<gridSize, blockSize, sharedMemorySize>>>(numRuns, runRank, stateStartDevice, jumpAheadMatricesDevice, numSteps, results);
        error = cudaGetLastError();
        if (error != cudaSuccess)
            return error;

		// the old code doesn't sync thread here
		error = cudaThreadSynchronize();
        if (error != cudaSuccess)
            return error;

		clock_t t1 = clock();
		printf(" [%f] ", (double(t1 - t0) / CLK_TCK));

        // Free device memory used.
        error = cudaFree(jumpAheadMatricesDevice);
        if (error != cudaSuccess)
            return error;
        error = cudaFree(stateStartDevice);
        if (error != cudaSuccess)
            return error;
        
        return error;
    }

	cudaError_t rngGenerateUniformKernelExecuteUnsignedWithTimeLogging(
        int         numRuns
      , int         runRank
      , dim3        gridSize
      , dim3        blockSize
      , unsigned    seed
      , int         numSteps
      , unsigned  * results
	  , bool		syncThread
      )
    {
        cudaError_t error;
		clock_t t0, t1;

        // Generate RNG initial state, and copy it to the device memory.
		t0 = clock();
        unsigned stateStart[8];
        rngGetStartState(seed, stateStart);
        unsigned* stateStartDevice;
        error = cudaMalloc((void**) &stateStartDevice, 8 * sizeof(unsigned));
        if (error != cudaSuccess)
            return error;
        error = cudaMemcpy(stateStartDevice, stateStart, sizeof(stateStart), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
            return error;
		t1 = clock();
		//printf(".. prepare init state (%d bytes) : %f seconds\n", sizeof(stateStart), ((double(t1) - double(t0)) / CLOCKS_PER_SEC));
		

        // Copy pre-calculated bit-matrices, needed for jump-ahead
        // calculations, to the device memory.
		t0 = clock();
        unsigned* jumpAheadMatricesDevice;
        error = cudaMalloc((void**) &jumpAheadMatricesDevice, sizeof(xorshift7JumpAheadMatrices));
        if (error != cudaSuccess)
            return error;
        error = cudaMemcpy(jumpAheadMatricesDevice, xorshift7JumpAheadMatrices, sizeof(xorshift7JumpAheadMatrices), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
            return error;
		t1 = clock();
		//printf(".. prepare jump ahead matrices (%d bytes) : %f seconds\n", sizeof(xorshift7JumpAheadMatrices), ((double(t1) - double(t0)) / CLOCKS_PER_SEC));

        // Calculate shared memory size needed, and then execute the kernel.
        // Besides using the shared memory for the RNG data structure, it
        // will be used as cache for jump-ahead matrix in the corresponding
        // device function above.  Thus, the shared memory size should be at
        // least the same as the jump-ahead matrix row size; however, as the
        // size of this row is the same as the RNG state size, and as the
        // block must contain at least one thread, this requirement will be
        // always satisfied.
        int sharedMemorySize = blockSize.x * blockSize.y * blockSize.z * xorshift7GetSize();
		//printf(".. launch kernel:\n");
		//printf("..   syncThread: %d\n", int(syncThread));
		//printf("..   shape: grid(%dx%dx%d) block(%dx%dx%d)\n", gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z);
		//printf("..   numRuns(%d) runRank(%d) numSteps(%d)\n", numRuns, runRank, numSteps);
		//printf("..   sharedMemorySize: %d bytes\n", sharedMemorySize);

		t0 = clock();
        xorshift7GenerateUniformKernelUnsigned<<<gridSize, blockSize, sharedMemorySize>>>(numRuns, runRank, stateStartDevice, jumpAheadMatricesDevice, numSteps, results);
        error = cudaGetLastError();
        if (error != cudaSuccess)
            return error;
		if (syncThread)
		{
			// the old code doesn't sync thread here
			error = cudaThreadSynchronize();
			if (error != cudaSuccess)
				return error;
		}
		t1 = clock();
		//printf(".. 1st launch kernel: %f seconds\n", ((double(t1) - double(t0)) / CLOCKS_PER_SEC));
		printf("cuda: %f\n", (double(t1) - double(t0)) / double(CLOCKS_PER_SEC));

		//t0 = clock();
  //      xorshift7GenerateUniformKernelUnsigned<<<gridSize, blockSize, sharedMemorySize>>>(numRuns, runRank, stateStartDevice, jumpAheadMatricesDevice, numSteps, results);
  //      error = cudaGetLastError();
  //      if (error != cudaSuccess)
  //          return error;
		//if (syncThread)
		//{
		//	// the old code doesn't sync thread here
		//	error = cudaThreadSynchronize();
		//	if (error != cudaSuccess)
		//		return error;
		//}
		//t1 = clock();
		//printf(".. 2nd launch kernel: %f seconds\n", ((double(t1) - double(t0)) / CLOCKS_PER_SEC));

		//t0 = clock();
  //      xorshift7GenerateUniformKernelUnsigned<<<gridSize, blockSize, sharedMemorySize>>>(numRuns, runRank, stateStartDevice, jumpAheadMatricesDevice, numSteps, results);
  //      error = cudaGetLastError();
  //      if (error != cudaSuccess)
  //          return error;
		//if (syncThread)
		//{
		//	// the old code doesn't sync thread here
		//	error = cudaThreadSynchronize();
		//	if (error != cudaSuccess)
		//		return error;
		//}
		//t1 = clock();
		//printf(".. 3rd launch kernel: %f seconds\n", ((double(t1) - double(t0)) / CLOCKS_PER_SEC));

		t0 = clock();
        // Free device memory used.
        error = cudaFree(jumpAheadMatricesDevice);
        if (error != cudaSuccess)
            return error;
        error = cudaFree(stateStartDevice);
        if (error != cudaSuccess)
            return error;
		t1 = clock();
		//printf(".. finalize: %f seconds\n", ((double(t1) - double(t0)) / CLOCKS_PER_SEC));
        
        return error;
    }

	cudaError_t rngGenerateNormalKernelExecute(
        int         numRuns
      , int         runRank
      , dim3        gridSize
      , dim3        blockSize
      , unsigned    seed
      , int         numSteps
      , float     * results
      )
    {
        cudaError_t error;

        // Generate RNG initial state, and copy it to the device memory.
        unsigned stateStart[8];
        rngGetStartState(seed, stateStart);
        unsigned* stateStartDevice;
        error = cudaMalloc((void**) &stateStartDevice, 8 * sizeof(unsigned));
        if (error != cudaSuccess)
            return error;
        error = cudaMemcpy(stateStartDevice, stateStart, sizeof(stateStart), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
            return error;

        // Copy pre-calculated bit-matrices, needed for jump-ahead
        // calculations, to the device memory.
        unsigned* jumpAheadMatricesDevice;
        error = cudaMalloc((void**) &jumpAheadMatricesDevice, sizeof(xorshift7JumpAheadMatrices));
        if (error != cudaSuccess)
            return error;
        error = cudaMemcpy(jumpAheadMatricesDevice, xorshift7JumpAheadMatrices, sizeof(xorshift7JumpAheadMatrices), cudaMemcpyHostToDevice);
        if (error != cudaSuccess)
            return error;

        // Calculate shared memory size needed, and then execute the kernel.
        // Besides using the shared memory for the RNG data structure, it
        // will be used as cache for jump-ahead matrix in the corresponding
        // device function above.  Thus, the shared memory size should be at
        // least the same as the jump-ahead matrix row size; however, as the
        // size of this row is the same as the RNG state size, and as the
        // block must contain at least one thread, this requirement will be
        // always satisfied.
        int sharedMemorySize = blockSize.x * blockSize.y * blockSize.z * xorshift7GetSize();
        xorshift7GenerateNormalKernel<<<gridSize, blockSize, sharedMemorySize>>>(numRuns, runRank, stateStartDevice, jumpAheadMatricesDevice, numSteps, results);
        error = cudaGetLastError();
        if (error != cudaSuccess)
            return error;

        // Free device memory used.
        error = cudaFree(jumpAheadMatricesDevice);
        if (error != cudaSuccess)
            return error;
        error = cudaFree(stateStartDevice);
        if (error != cudaSuccess)
            return error;
        
        return error;
    }

}}}}
