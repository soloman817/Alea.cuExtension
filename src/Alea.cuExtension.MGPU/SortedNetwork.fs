module Alea.cuExtension.MGPU.SortedNetwork
// PARTIAL IMPLEMENTATION, NEEDS TESTING
open System.Runtime.InteropServices
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.cuExtension
//open Alea.cuExtension.Util
open Alea.cuExtension.MGPU
//open Alea.cuExtension.MGPU.QuotationUtil
open Alea.cuExtension.MGPU.DeviceUtil
open Alea.cuExtension.MGPU.LoadStore
open Alea.cuExtension.MGPU.CTAScan

//
//
////////////////////////////////////////////////////////////////////////////////////
////// Odd-even transposition sorting network. Sorts keys and values in-place in
////// register.
////// http://en.wikipedia.org/wiki/Odd%E2%80%93even_sort
//type IOddEvenTransposeSortT<int> =
//    abstract member Sort : Expr<deviceptr<int> -> deviceptr<int> -> int -> unit>
//
////let inline oddEvenTransposeSortT (I:int) (VT:int) (compOp:IComp<int>) =
////    { new IOddEvenTransposeSortT<int> with
////        member this.Sort = 
////            let swap = (swap compOp.Identity).Device
////            let comp = compOp.Device            
////            <@ fun (keys:deviceptr<int>) (values:deviceptr<int>) (flags:int)  ->
////                let swap = %swap
////                let comp = %comp
////                            
////                let mutable i = 1 &&& I
////                while i < VT - 1 do
////                    if (((2 <<< i) &&& flags) = 0) && (comp keys.[i + 1] keys.[i]) then
////                        swap keys.[i] keys.[i + 1]
////                        swap values.[i] values.[i + 1]
////                    i <- i + 2                
////                 @> }
//
//
//let oddEvenTransposeSortT (VT:int) (compOp:IComp<int>) =
//    let comp = compOp.Device        
//    <@ fun (keys:deviceptr<int>) (values:deviceptr<int>) (flags:int) ->        
//        let comp = %comp
//        let mutable level = 0
//        while level < VT do
//            let mutable i = 1 &&& level
//            while i < VT - 1 do
//                //if (((2 <<< i) &&& flags) = 0) && (comp keys.[i + 1] keys.[i]) then
//                if ( comp keys.[i + 1] keys.[i] ) then
//                    swap keys.[i] keys.[i + 1]
//                    swap values.[i] values.[i + 1]
//                i <- i + 2  
//            level <- level + 1 
//    @>
//
//
//
//
////let OddEvenTransposeSortT (I:int) (VT:int) (compOp:IComp<'TV>) =
////    let swap = (swap compOp.Identity).Device
////    let oddEvenTransposeSortT = (oddEvenTransposeSortT (I + 1) VT compOp).Sort
////    let comp = compOp.Device
////    <@ fun (keys:deviceptr<'TV>) (values:deviceptr<'TV>) (flags:int) ->
////        let swap = %swap
////        let comp = %comp
////        let oddEvenTransposeSortT = %oddEvenTransposeSortT
////
////        let mutable i = 1 &&& I
////        while i < VT - 2 do
////            if (((2 <<< i) &&& flags) = 0) && (comp keys.[i + 1] keys.[i]) then
////                swap keys.[i] keys.[i + 1]
////                swap values.[i] values.[i + 1]
////            i <- i + 2
////        oddEvenTransposeSortT keys values flags
////            @>
//
//
//let oddEvenTransposeSort (VT:int) (compOp:IComp<'TV>) =
//    let oddEvenTransposeSortT = oddEvenTransposeSortT VT compOp
//    <@ fun (keys:deviceptr<'TV>) (values:deviceptr<'TV>) ->
//        let oddEvenTransposeSortT = %oddEvenTransposeSortT
//        oddEvenTransposeSortT keys values 0
//    @>
//
//
//let oddEvenTransposeSortFlags (VT:int) (compOp:IComp<'TV>) =
//    <@ fun (keys:deviceptr<'TV>) (values:deviceptr<'TV>) (flags:int) ->
//        ()
//    @>
//
//let oddEvenMergesortFlags (VT:int) (compOp:IComp<'TV>) =
//    <@ fun (keys:deviceptr<'TV>) (values:deviceptr<'TV>) (flags:int) ->
//        ()
//    @>
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
