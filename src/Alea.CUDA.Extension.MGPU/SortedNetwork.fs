﻿module Alea.CUDA.Extension.MGPU.SortedNetwork

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTAScan





//////////////////////////////////////////////////////////////////////////////////
//// Odd-even transposition sorting network. Sorts keys and values in-place in
//// register.
//// http://en.wikipedia.org/wiki/Odd%E2%80%93even_sort
//	
//// CUDA Compiler does not currently unroll these loops correctly. Write using
//// template loop unrolling.
///*
//template<int VT, typename T, typename V, typename Comp>
//MGPU_DEVICE void OddEvenTransposeSort(T* keys, V* values, Comp comp) {
//	#pragma unroll
//	for(int level = 0; level < VT; ++level) {
//
//		#pragma unroll
//		for(int i = 1 & level; i < VT - 1; i += 2) {
//			if(comp(keys[i + 1], keys[i])) {
//				mgpu::swap(keys[i], keys[i + 1]);
//				mgpu::swap(values[i], values[i + 1]);
//			}
//		}
//	}
//}*/
//
//template<int I, int VT>
//struct OddEvenTransposeSortT {
//	// Sort segments marked by head flags. If the head flag between i and i + 1
//	// is set (so that (2<< i) & flags is true), the values belong to different 
//	// segments and are not swapped.
//	template<typename K, typename V, typename Comp>
//	static MGPU_DEVICE void Sort(K* keys, V* values, int flags, Comp comp) {
//		#pragma unroll
//		for(int i = 1 & I; i < VT - 1; i += 2)
//			if((0 == ((2<< i) & flags)) && comp(keys[i + 1], keys[i])) {
//				mgpu::swap(keys[i], keys[i + 1]);
//				mgpu::swap(values[i], values[i + 1]);
//			}
//		OddEvenTransposeSortT<I + 1, VT>::Sort(keys, values, flags, comp);
//	}
//};
//template<int I> struct OddEvenTransposeSortT<I, I> {
//	template<typename K, typename V, typename Comp>
//	static MGPU_DEVICE void Sort(K* keys, V* values, int flags, Comp comp) { }
//};
//
//template<int VT, typename K, typename V, typename Comp>
//MGPU_DEVICE void OddEvenTransposeSort(K* keys, V* values, Comp comp) {
//	OddEvenTransposeSortT<0, VT>::Sort(keys, values, 0, comp);
//}
//template<int VT, typename K, typename V, typename Comp>
//MGPU_DEVICE void OddEvenTransposeSortFlags(K* keys, V* values, int flags, 
//	Comp comp) {
//	OddEvenTransposeSortT<0, VT>::Sort(keys, values, flags, comp);
//}
//
//////////////////////////////////////////////////////////////////////////////////
//// Batcher Odd-Even Mergesort network
//// Unstable but executes much faster than the transposition sort.
//// http://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
//
//template<int Width, int Low, int Count>
//struct OddEvenMergesortT {
//	template<typename K, typename V, typename Comp>
//	MGPU_DEVICE static void CompareAndSwap(K* keys, V* values, int flags,
//		int a, int b, Comp comp) {
//		if(b < Count) {
//			// Mask the bits between a and b. Any head flags in this interval
//			// means the keys are in different segments and must not be swapped.
//			const int Mask = ((2<< b) - 1) ^ ((2<< a) - 1);
//			if(!(Mask & flags) && comp(keys[b], keys[a])) {
//				mgpu::swap(keys[b], keys[a]);
//				mgpu::swap(values[b], values[a]);
//			}
//		}
//	}
//
//	template<int R, int Low2, bool Recurse = 2 * R < Width>
//	struct OddEvenMerge {
//		template<typename K, typename V, typename Comp>
//		MGPU_DEVICE static void Merge(K* keys, V* values, int flags,
//			Comp comp) {
//			// Compare and swap
//			const int M = 2 * R;
//			OddEvenMerge<M, Low2>::Merge(keys, values, flags, comp);
//			OddEvenMerge<M, Low2 + R>::Merge(keys, values, flags, comp);
//		
//			#pragma unroll
//			for(int i = Low2 + R; i + R < Low2 + Width; i += M)
//				CompareAndSwap(keys, values, flags, i, i + R, comp);
//		}
//	};
//	template<int R, int Low2> 
//	struct OddEvenMerge<R, Low2, false> {
//		template<typename K, typename V, typename Comp>
//		MGPU_DEVICE static void Merge(K* keys, V* values, int flags,
//			Comp comp) {
//			CompareAndSwap(keys, values, flags, Low2, Low2 + R, comp);
//		}
//	};
//	
//	template<typename K, typename V, typename Comp>
//	MGPU_DEVICE static void Sort(K* keys, V* values, int flags,
//		Comp comp) {
//
//		const int M = Width / 2;
//		OddEvenMergesortT<M, Low, Count>::Sort(keys, values, flags, comp);
//		OddEvenMergesortT<M, Low + M, Count>::Sort(keys, values, flags, comp);
//		OddEvenMerge<1, Low>::Merge(keys, values, flags, comp);
//	}
//};
//template<int Low, int Count> struct OddEvenMergesortT<1, Low, Count> {
//	template<typename K, typename V, typename Comp>
//	MGPU_DEVICE static void Sort(K* keys, V* values, int flags,
//		Comp comp) { }
//};
//
//template<int VT, typename K, typename V, typename Comp>
//MGPU_DEVICE void OddEvenMergesort(K* keys, V* values, Comp comp) {
//	const int Width = 1<< sLogPow2<VT, true>::value;
//	OddEvenMergesortT<Width, 0, VT>::Sort(keys, values, 0, comp);
//}
//template<int VT, typename K, typename V, typename Comp>
//MGPU_DEVICE void OddEvenMergesortFlags(K* keys, V* values, int flags,
//	Comp comp) {
//	const int Width = 1<< sLogPow2<VT, true>::value;
//	OddEvenMergesortT<Width, 0, VT>::Sort(keys, values, flags, comp);
//}
//
//} // namespace mgpu
