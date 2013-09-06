#include "xorshift7_kernels.hpp"
#include "xorshift7_gold.hpp"
#include "xorshift7_jump_ahead_matrices.hpp"
#include "xorshift7_jump_ahead.hpp"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <cuda.h>

using namespace alea::cuda::math::random;

static unsigned int x[8];
static int k = 0;

void initxorshift7(unsigned int* init)
{
	for (int j=0; j<8; ++j) x[j] = init[j];
}

unsigned xorshift7Unsigned(void)
{
	unsigned int y, t;
	t = x[(k+7) & 0x7U];   t = t ^ (t<<13);   y = t ^ (t<<9);
	t = x[(k+4) & 0x7U];   y^= t ^ (t<<7);
	t = x[(k+3) & 0x7U];   y^= t ^ (t>>3);
	t = x[(k+1) & 0x7U];   y^= t ^ (t>>10);
	t = x[k];              t = t ^ (t>>7);    y^= t ^ (t<<24);
	x[k] = y;              k = (k+1) & 0x7U;
	return y;
}

double xorshift7()
{
	return (double)xorshift7Unsigned() * 2.32830643653869628906e-10;
}

void test1() 
{
	unsigned seed = 0;

	unsigned state[8];
	Xorshift7Gold rng(seed);
	rng.getState(state);
	initxorshift7(state);
	k = 0;

	int error = 0;
	for (int i = 0; i < 100; ++i) 
	{
		rng.getState(state);
		for (int j = 0; j < 8; ++j)
			std::cout << "state " << j << " : gold " << x[(k+j) & 0x7] << ", rng " << state[j] << std::endl;
		unsigned r1 = xorshift7Unsigned();
		unsigned r2 = rng.getUnsignedUniform();
		std::cout << "xorshift7() = " << r1 << ", rng.getUnsignedUniform() = " << r2 << std::endl;
		error += r1 == r2 ? 0 : 1;
	}
	std::cout << "error = " << error << std::endl;

	for (int i = 0; i < 100; ++i)
		std::cout << std::setprecision(14) << "xorshift7() = " << xorshift7() << ", rng.getFloatUniform() = " << rng.getFloatUniform() << std::endl;
}

void printMat(const Matrix256 & m)
{
	for (int i = 0; i < 256; ++i)
	{
		for (int j = 0; j < 8; ++j)
			std::cout << std::hex << std::setw(8) << m.bits_[i][j] << ", ";
		std::cout << std::endl;
	}
}

void test2()
{
	unsigned seed = 0;

	Xorshift7Gold rng(seed);

	Matrix256 m = Xorshift7Gold::getMatrix();

	//printMat(m);

	unsigned state[8];
	rng.getState(state);
	Vector256 v(state);

	//Vector256 vc = v;

	for (int i = 1; i <= 64; ++i)
	{
		rng.getUnsignedUniform();
		Matrix256 mi = m.pow(i);
		Vector256 vc = mi * v;
		rng.getState(state);
		bool agree = (Vector256(state) == vc);
		std::cout << i << ": agree=" << agree << std::endl;
	}
}

void cudaSafeCall(cudaError_t error)
{
	if (error != CUDA_SUCCESS)
	{
		throw std::exception();
	}
}

// Copied from an old test case of Alex

// This program is dumping, to its standard output, a set of sequences
// of xorshift7 RNG generated and uniformly distributed real random
// numbers (with the purpose of using these sequences later for unit
// testing CUDA xorshift7 RNG implementation).  The type of distribution
// (0 for uniform float, 1 for uniform unsigned, 2 for normal), the seed used to generate initial
// state of the xorshift7 RNG (standard C library RNG is seeded with
// this value, and then used to generate initial xorshift7 RNG state -
// it is supposed that the CUDA implementation will use the same method
// of initializing xorshift7 RNG sate), as well as requested number of
// sequences, and sequence length, are supplied in command line.  The
// program is first dumping the type of distribution and the seed, then
// number of sequences and sequence length, and then the sequences
// themselves (all numbers from first sequence first, then all numbers
// from second sequence, etc.).
int dumper(int argc, char** argv)
{
	// Verify that proper number of arguments is supplied in command
	// line
	assert(argc == 5);

	std::istringstream stream;

	// Parse the type of distribution from the command line.
	unsigned distribution = -1;
	stream.clear();
	stream.str(argv[1]);
	stream >> distribution;
	assert(distribution == 0 || distribution == 1 || distribution == 2);

	// Parse the seed value from the command line.
	unsigned seed = 0;
	stream.clear();
	stream.str(argv[2]);
	stream >> seed;

	// Parse number of sequences from the command line.
	int numSequences = 0;
	stream.clear();
	stream.str(argv[3]);
	stream >> numSequences;
	assert(numSequences > 0);

	// Parse sequence length from the command line.
	int numSteps = 0;
	stream.clear();
	stream.str(argv[4]);
	stream >> numSteps;
	assert(numSteps > 0);

	// Calculate jump-ahead matrix, using first power of 2 greater of
	// equal to the number of sequences.
	int p = 0;
	for ( ; (1 << p) < numSequences; ++p)
		;
	Matrix256 m = Xorshift7Gold::getMatrix().powPow2(256 - p);

	// Print type of distribution, seed, number of sequences, and
	// sequence length, to the standard output.
	std::cout << distribution << "\n";
	std::cout << seed << "\n";
	std::cout << numSequences << "\n";
	std::cout << numSteps << "\n";

	// Get initial state of xorshift7 RNG with given seed.
	unsigned state[8];
	Xorshift7Gold rng(seed);
	rng.getState(state);

	// Generate requested sequence of uniformly distributed random
	// numbers, and print them to the standard output.
	Vector256 v(state);
	for (int i = 0; i < numSequences; ++i) {
		// Print the current sequence.
		Xorshift7Gold rng(v.bits_);
		switch (distribution) {
		case 0:
			for (int j = 0; j < numSteps; ++j) 
				std::cout << rng.getFloatUniform() << "\n";
			break;

		case 1:
			for (int j = 0; j < numSteps; ++j) 
				std::cout << rng.getUnsignedUniform() << "\n";
			break;

		case 2:
			{
				float value0, value1;
				int count = numSteps / 2;
				for (int i = 0; i < count; ++i) {
					rng.getFloatNormal(&value0, &value1); 
					std::cout << value0 << "\n";
					std::cout << value1 << "\n";
				}
				if (numSteps & 0x1) {
					rng.getFloatNormal(&value0, &value1);
					std::cout << value0 << "\n";
				}
			}
			break;
		}

		// Jump ahead to the start of the next sequence.
		v = m * v;
	}

	return 0;
}

#include <time.h>
#include <stdio.h>

void test3()
{
	cudaSafeCall(cudaSetDevice(0));

	unsigned seed = 42u;

	int numDevices = 1;
	dim3 gridSize(448, 1, 1);
	dim3 blockSize(32, 8, 1);
	int numSteps = 5;

	int numThreadsPerBlock = blockSize.x * blockSize.y * blockSize.z;
	int numBlocksPerDevice = gridSize.x * gridSize.y * gridSize.z;
	int numThreadsPerDevice = numThreadsPerBlock * numBlocksPerDevice;
	int numSequence = numThreadsPerDevice * numDevices;
	int numTotal = numSequence * numSteps;

	std::cout << "seed = " << seed << std::endl;
	std::cout << "numSequence = " << numSequence << std::endl;
	std::cout << "numSteps = " << numSteps << std::endl;

	std::vector<unsigned> results(numTotal);

	unsigned* dResults;
	cudaSafeCall(cudaMalloc(&dResults, numThreadsPerDevice * numSteps * sizeof(unsigned)));

	for (int d = 0; d < numDevices; ++d)
	{
		{
			std::cout << "launch kernel on device " << d << "...";
			clock_t begin = clock();
			cudaSafeCall(rngGenerateUniformKernelExecuteUnsigned(
				numDevices, d, gridSize, blockSize, seed, numSteps, dResults));
			clock_t end = clock();
			clock_t span = end - begin;
			std::cout << "[OK] " << (double(span) / CLK_TCK) << std::endl;
		}

		//{
		//	std::cout << "launch kernel on device " << d << "...";
		//	clock_t begin = clock();
		//	cudaSafeCall(rngGenerateUniformKernelExecuteUnsigned(
		//		numDevices, d, gridSize, blockSize, seed, numSteps, dResults));
		//	clock_t end = clock();
		//	clock_t span = end - begin;
		//	std::cout << "[OK] " << (double(span) / CLK_TCK) << std::endl;
		//}

		//{
		//	std::cout << "launch kernel on device " << d << "...";
		//	clock_t begin = clock();
		//	cudaSafeCall(rngGenerateUniformKernelExecuteUnsigned(
		//		numDevices, d, gridSize, blockSize, seed, numSteps, dResults));
		//	clock_t end = clock();
		//	clock_t span = end - begin;
		//	std::cout << "[OK] " << (double(span) / CLK_TCK) << std::endl;
		//}



		std::vector<unsigned> hResults(numThreadsPerDevice * numSteps);
		cudaSafeCall(cudaMemcpy(&hResults[0], dResults, numThreadsPerDevice * numSteps * sizeof(unsigned), cudaMemcpyDeviceToHost));

		for (int i = 0; i < 10; ++i)
		{
			std::cout << hResults[i] << " ";
		}
		std::cout << std::endl;

		// fill the global result
		for (int s = 0; s < numSteps; ++s)
		{
			std::copy(
				hResults.begin() + s * numThreadsPerDevice,
				hResults.begin() + (s + 1) * numThreadsPerDevice,
				results.begin() + s * numSequence + d * numThreadsPerDevice);
		}
	}

	cudaSafeCall(cudaFree(dResults));

	// Get initial state of xorshift7 RNG with given seed.
	unsigned state_[8];
	Xorshift7Gold rng_(seed);
	rng_.getState(state_);

	// Calculate jump-ahead matrix, using first power of 2 greater of
	// equal to the number of sequences.
	int p = 0;
	for (; (1 << p) < numSequence; ++p);
	std::cout << "p = " << p << std::endl;
	Matrix256 m;
	bool useGenerated = false;
	if (!useGenerated)
	{
		int lo = 224;
		int tableid = 256 - p - lo;
		const unsigned* table = &xorshift7JumpAheadMatrices[tableid * 256 * 8];
		for (int i = 0; i < 256; ++i)
			for (int j = 0; j < 8; ++j)
				m.bits_[i][j] = table[i * 8 + j];
	}
	else
	{
		m = Xorshift7Gold::getMatrix().powPow2(256 - p);
	}
	//printMat(m);
	
	int error = 0;
	Vector256 v(state_);
	for (int i = 0; i < numSequence; ++i) 
	{
		Xorshift7Gold rng(v.bits_);
		initxorshift7(v.bits_);
		k = 0;

		for (int s = 0; s < numSteps; ++s)
		{
			unsigned rMe = xorshift7Unsigned();
			unsigned rGold = rng.getUnsignedUniform();
			unsigned rCuda = results[s * numSequence + i];
			if ((rMe == rGold) && (rMe == rCuda))
			{
				//std::cout << "     #" << i << "." << s << ": me(" << rMe << ") gold(" << rGold << ") cuda(" << rCuda << ")" << std::endl;
			}
			else
			{
				++error;
				std::cout << "!!!! #" << i << "." << s << ": me(" << rMe << ") gold(" << rGold << ") cuda(" << rCuda << ")" << std::endl;
			}
		}

		// jump ahead
		v = m * v;
	}
	std::cout << "error = " << error << std::endl;
}

void testPerformance(int numSteps)
{
	cudaSafeCall(cudaSetDevice(0));

	unsigned seed = 42u;
	dim3 gridSize(8, 1, 1);
	dim3 blockSize(32, 8, 1);
	int numDevices = 1;

	int numThreadsPerBlock = blockSize.x * blockSize.y * blockSize.z;
	int numBlocksPerDevice = gridSize.x * gridSize.y * gridSize.z;
	int numThreadsPerDevice = numThreadsPerBlock * numBlocksPerDevice;
	int numStreams = numThreadsPerDevice * numDevices;

	std::cout << "=======================================" << std::endl;
	std::cout << "numStreams = " << numStreams << std::endl;
	std::cout << "numSteps = " << numSteps << std::endl;

	std::vector<unsigned> resultsCuda(numStreams * numSteps);

	int d = 0;

	unsigned* dResults;
	cudaSafeCall(cudaMalloc(&dResults, numThreadsPerDevice * numSteps * sizeof(unsigned)));
	double secondsCuda = 1e10;

	cudaSafeCall(rngGenerateUniformKernelExecuteUnsignedWithTimeLogging(
		numDevices, d, gridSize, blockSize, seed, numSteps, dResults, true));

	cudaSafeCall(rngGenerateUniformKernelExecuteUnsignedWithTimeLogging(
		numDevices, d, gridSize, blockSize, seed, numSteps, dResults, true));

	cudaSafeCall(rngGenerateUniformKernelExecuteUnsignedWithTimeLogging(
		numDevices, d, gridSize, blockSize, seed, numSteps, dResults, true));

	std::vector<unsigned> hResults(numThreadsPerDevice * numSteps);
	cudaSafeCall(cudaMemcpy(&hResults[0], dResults, numThreadsPerDevice * numSteps * sizeof(unsigned), cudaMemcpyDeviceToHost));

	// fill the global result
	for (int i = 0; i < numSteps; ++i)
	{
		std::copy(
			hResults.begin() + i * numThreadsPerDevice,
			hResults.begin() + (i + 1) * numThreadsPerDevice,
			resultsCuda.begin() + i * numStreams + d * numThreadsPerDevice);
	}

	cudaSafeCall(cudaFree(dResults));

	// Get initial state of xorshift7 RNG with given seed.
	unsigned state_[8];
	Xorshift7Gold rng_(seed);
	rng_.getState(state_);

	// compute with C
	// Calculate jump-ahead matrix, using first power of 2 greater of
	// equal to the number of sequences.
	int p = 0;
	for (; (1 << p) < numStreams; ++p);
	Matrix256 m;
	bool useGenerated = false;
	if (!useGenerated)
	{
		int lo = 224;
		int tableid = 256 - p - lo;
		const unsigned* table = &xorshift7JumpAheadMatrices[tableid * 256 * 8];
		for (int i = 0; i < 256; ++i)
			for (int j = 0; j < 8; ++j)
				m.bits_[i][j] = table[i * 8 + j];
	}
	else
	{
		m = Xorshift7Gold::getMatrix().powPow2(256 - p);
	}
	
	clock_t t0, t1;
	t0 = clock();
	std::vector<unsigned> resultsGold(numStreams * numSteps);
	Vector256 v(state_);
	for (int s = 0; s < numStreams; ++s)
	{
		Xorshift7Gold rng(v.bits_);
		for (int i = 0; i < numSteps; ++i)
			resultsGold[i * numStreams + s] = rng.getUnsignedUniform();
		v = m * v;
	}
	t1 = clock();
	double secondsGold = ((double(t1) - double(t0)) / CLOCKS_PER_SEC);

	std::cout << "gold: " << secondsGold << std::endl;

	// verify
	int error = 0;
	v = Vector256(state_);
	for (int s = 0; s < numStreams; ++s) 
	{
		initxorshift7(v.bits_);
		k = 0;
		for (int i = 0; i < numSteps; ++i)
		{
			unsigned rMe = xorshift7Unsigned();
			unsigned rGold = resultsGold[i * numStreams + s];
			unsigned rCuda = resultsCuda[i * numStreams + s];
			if ((rMe == rGold) && (rMe == rCuda))
			{
				//std::cout << "     #" << i << "." << s << ": me(" << rMe << ") gold(" << rGold << ") cuda(" << rCuda << ")" << std::endl;
			}
			else
			{
				++error;
				std::cout << "!!!! #" << i << "." << s << ": me(" << rMe << ") gold(" << rGold << ") cuda(" << rCuda << ")" << std::endl;
			}
		}
		v = m * v;
	}
	std::cout << "error = " << error << std::endl;
	std::cout << "=======================================" << std::endl;
}

int main(int argc, char** argv)
{
	//test1();
	//test2();
	//test3();
	testPerformance(1000);
	testPerformance(5000);
	testPerformance(10000);
	testPerformance(20000);
	testPerformance(30000);
	testPerformance(40000);
	testPerformance(50000);
	//dumper(argc, argv);
	//xorshift7JumpHeaderGenerateFSharp(224, 255);
	return 0;
}