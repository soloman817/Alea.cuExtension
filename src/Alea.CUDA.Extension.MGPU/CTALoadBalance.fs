module Alea.CUDA.Extension.MGPU.CTALoadBalance

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTASearch




////////////////////////////////////////////////////////////////////////////////
// DeviceLoadBalancingSearch
// Upper Bound search from A (needles) into B (haystack). The A values are 
// natural numbers from aBegin to aEnd. bFirst is the index of the B value at
// bBegin in shared memory.

//template<int VT, bool RangeCheck>
//let deviceSerialLoadBalanceSearch (VT:int) (rangeCheck:int)
//    <@ fun (b_shared:RWPtr<int>) (aBegin:int) (aEnd:int) (bFirst:int) (bBegin:int) (bEnd:int) (a_shared:RWPtr<int>) -> 
//        let mutable bKey = b_shared.[bBegin]
//        for i = 0 to VT - 1 do
//		    let mutable p = 0
//		    if rangeCheck = 1 then
//			    p <- if (aBegin < aEnd) && ((bBegin >= bEnd) || (aBegin < bKey)) then 1 else 0
//		    else
//			    p <- if aBegin < bKey then 1 else 0
//
//		    if p = 1 then
//			    // Advance A (the needle).
//			    a_shared.[aBegin++] <- bFirst + bBegin
//		    else
//			    // Advance B (the haystack).
//			    bKey <- b_shared.[++bBegin] @>
//
//
//// CTALoadBalance
//let ctaLoadBalance (NT:int) (VT:int) =
//    
//
//    let loadBalance =
//        <@ fun (destCount:int) (b_global:RWPtr<int>) (sourceCount:int) (block:int) (tid:int) (mp_global:RWPtr<int>) (indices_shared:RWPtr<int>) (loadPrecedingB:int) ->
//            
//            let range = computMergeRange destCount sourceCount block 0 (NT * VT) mp_global
//
//            let a0 = range.x
//            let a1 = range.y
//            let mutable b0 = range.z
//            let b1 = range.w
//
//            let mutable loadPrecedingB = loadPrecedingB
//            if loadPrecedingB = 1 then
//                if b0 <> 0 then
//                    loadPrecedingB = 0
//                else
//                    b0 <- b0 - 1
//            
//            let mutable extended = if (a1 < destCount) && (b1 < sourceCount) then 1 else 0
//
//            let mutable aCount = a1 - a0
//            let mutable bCount = b1 - b0
//
//            let a_shared = indices_shared
//            let b_shared = indices_shared + aCount
//
//            deviceMemToMemLoop (bCount + extended) (b_global + b0) tid b_shared
//
//            let diag = min((VT * tid) (aCount + bCount - loadPrecedingB))
////            int mp = MergePath<MgpuBoundsUpper>(mgpu::counting_iterator<int>(a0),
////		        aCount, b_shared + (int)loadPrecedingB, bCount - (int)loadPrecedingB,
////		        diag, mgpu::less<int>());
//
//            let a0tid = a0 + mp
//            let b0tid = diag - mp + loadPrecedingB
//
//
//            if extended = 1 then
//                ()
////                DeviceSerialLoadBalanceSearch<VT, false>(b_shared, a0tid, a1, b0 - 1,
////			        b0tid, bCount, a_shared - a0);
//            else
////                DeviceSerialLoadBalanceSearch<VT, true>(b_shared, a0tid, a1, b0 - 1, 
////			        b0tid, bCount, a_shared - a0);
//            __syncthreads()
//
//            int4(a0, a1, b0, b1)
//             @>
//    