module Alea.cuExtension.MGPU.LocalitySort
// NOT IMPLEMENTED YET
open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.cuExtension
open Alea.cuExtension.Util
open Alea.cuExtension.MGPU
open Alea.cuExtension.MGPU.QuotationUtil
open Alea.cuExtension.MGPU.DeviceUtil
open Alea.cuExtension.MGPU.LoadStore
open Alea.cuExtension.MGPU.CTAScan




//namespace mgpu {
//
//template<typename T, typename Comp>
//MGPU_HOST void LocalitySortKeys(T* data_global, int count, CudaContext& context,
//	Comp comp, bool verbose) {
//
//	const int NT = 128;
//	const int VT = 11;
//	typedef LaunchBoxVT<NT, VT> Tuning;
//	int2 launch = Tuning::GetLaunchParams(context);
//	const int NV = launch.x * launch.y;
//
//	int numBlocks = MGPU_DIV_UP(count, NV);
//	int numPasses = FindLog2(numBlocks, true);
//
//	SegSortSupport support;
//	MGPU_MEM(byte) mem = AllocSegSortBuffers(count, NV, support, false,
//		context);
//	
//	MGPU_MEM(T) destDevice = context.Malloc<T>(count);
//	T* source = data_global;
//	T* dest = destDevice->get(); 
//	
//	KernelBlocksort<Tuning, false>
//		<<<numBlocks, launch.x, 0, context.Stream()>>>(source, (const int*)0,
//		count, (1 & numPasses) ? dest : source, (int*)0, comp);
//	if(1 & numPasses) std::swap(source, dest);
//
//	SegSortKeysPasses<Tuning, false>(support, source, count, numBlocks, 
//		numPasses, dest, comp, context, verbose);
//} 
//template<typename T>
//MGPU_HOST void LocalitySortKeys(T* data_global, int count, CudaContext& context,
//	bool verbose) {
//	LocalitySortKeys(data_global, count, context, mgpu::less<T>(), verbose);
//}
//
//template<typename KeyType, typename ValType, typename Comp>
//MGPU_HOST void LocalitySortPairs(KeyType* keys_global, ValType* values_global,
//	int count, CudaContext& context, Comp comp, bool verbose) {
//
//	const int NT = 128;
//	const int VT = 7;
//	typedef LaunchBoxVT<NT, VT> Tuning;
//	int2 launch = Tuning::GetLaunchParams(context);
//	const int NV = launch.x * launch.y;
//
//	int numBlocks = MGPU_DIV_UP(count, NV);
//	int numPasses = FindLog2(numBlocks, true);
//
//	SegSortSupport support;
//	MGPU_MEM(byte) mem = AllocSegSortBuffers(count, NV, support, false,
//		context);
//	
//	MGPU_MEM(KeyType) keysDestDevice = context.Malloc<KeyType>(count);
//	MGPU_MEM(ValType) valsDestDevice = context.Malloc<ValType>(count);
//
//	KeyType* keysSource = keys_global;
//	KeyType* keysDest = keysDestDevice->get();
//	ValType* valsSource = values_global;
//	ValType* valsDest = valsDestDevice->get();
//
//	KernelBlocksort<Tuning, true><<<numBlocks, launch.x, 0, context.Stream()>>>(
//		keysSource, valsSource, count, (1 & numPasses) ? keysDest : keysSource,
//		(1 & numPasses) ? valsDest : valsSource, comp);
//	if(1 & numPasses) {
//		std::swap(keysSource, keysDest);
//		std::swap(valsSource, valsDest);
//	}
//
//	SegSortPairsPasses<Tuning, false>(support, keysSource, valsSource, count,
//		numBlocks, numPasses, keysDest, valsDest, comp, context, verbose);
//} 
//template<typename KeyType, typename ValType>
//MGPU_HOST void LocalitySortPairs(KeyType* keys_global, ValType* values_global,
//	int count, CudaContext& context, bool verbose) {
//	LocalitySortPairs(keys_global, values_global, count, context,
//		mgpu::less<KeyType>(), verbose);
//}
//
//} // namespace mgpu