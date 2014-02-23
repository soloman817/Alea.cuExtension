[<AutoOpen>]
module Alea.cuExtension.MGPU.CTALoadBalance

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.cuExtension
//open Alea.cuExtension.Util
open Alea.cuExtension.MGPU
//open Alea.cuExtension.MGPU.QuotationUtil
open Alea.cuExtension.MGPU.DeviceUtil
open Alea.cuExtension.MGPU.LoadStore
open Alea.cuExtension.MGPU.CTASearch
open Alea.cuExtension.MGPU.CTAMerge

//
//
//
//////////////////////////////////////////////////////////////////////////////////
//// DeviceLoadBalancingSearch
//// Upper Bound search from A (needles) into B (haystack). The A values are 
//// natural numbers from aBegin to aEnd. bFirst is the index of the B value at
//// bBegin in shared memory.
//
//let deviceSerialLoadBalanceSearch (VT:int) (rangeCheck:bool) =
//    <@ fun (b_shared:deviceptr<int>) (aBegin:int) (aEnd:int) (bFirst:int) (bBegin:int) (bEnd:int) (a_shared:deviceptr<int>) -> 
//        let mutable bKey = b_shared.[bBegin]
//        let mutable aBegin = aBegin
//        let mutable bBegin = bBegin
//
//        for i = 0 to VT - 1 do
//            let mutable p = false
//            if rangeCheck then
//                p <- (aBegin < aEnd) && ((bBegin >= bEnd) || (aBegin < bKey))
//            else
//                p <- aBegin < bKey
//
//            if p then
//                // Advance A (the needle).
//                a_shared.[aBegin] <- bFirst + bBegin
//                aBegin <- aBegin + 1
//            else
//                // Advance B (the haystack).
//                bBegin <- bBegin + 1
//                bKey <- b_shared.[bBegin] @>
//
//
//////////////////////////////////////////////////////////////////////////////////
//// CTALoadBalance
//// Computes upper_bound(counting_iterator<int>(first), b_global) - 1.
//
//// Unlike most other CTA* functions, CTALoadBalance loads from global memory.
//// This returns the loaded B elements at the beginning or end of shared memory
//// depending on the aFirst argument. 
//
//// CTALoadBalance requires NT * VT + 2 slots of shared memory.
//let ctaLoadBalance (NT:int) (VT:int) =
//    let computeMergeRange = computeMergeRange.Device
//    let mergePath = (mergePath MgpuBoundsUpper (comp CompTypeLess 0)).DMergePath
//    let deviceMemToMemLoop = deviceMemToMemLoop NT
//    let deviceSerialLoadBalanceSearchFalse = deviceSerialLoadBalanceSearch VT false
//    let deviceSerialLoadBalanceSearchTrue = deviceSerialLoadBalanceSearch VT true
//    let itrSize = NT * (VT + 1)
//    let counting_iterator = counting_iterator itrSize
//    <@ fun  (destCount      :int) 
//            (b_global       :deviceptr<int>)
//            (sourceCount    :int)
//            (block          :int)
//            (tid            :int)
//            (countingItr_global :deviceptr<int>)
//            (mp_global      :deviceptr<int>)
//            (indices_shared :deviceptr<int>)
//            (loadPrecedingB :bool)
//            ->
//        let counting_iterator = %counting_iterator
//        let computeMergeRange = %computeMergeRange
//        let mergePath = %mergePath
//        let deviceMemToMemLoop = %deviceMemToMemLoop
//        let deviceSerialLoadBalanceSearchFalse = %deviceSerialLoadBalanceSearchFalse
//        let deviceSerialLoadBalanceSearchTrue = %deviceSerialLoadBalanceSearchTrue
//
//        let mutable loadPrecedingB = if loadPrecedingB then 1 else 0
//
//        let range = computeMergeRange destCount sourceCount block 0 (NT * VT) mp_global
//        
//        let a0 = range.x
//        let a1 = range.y
//        let mutable b0 = range.z
//        let b1 = range.w
//        
//        if loadPrecedingB = 1 then
//            if b0 = 0 then
//                loadPrecedingB <- 0
//            else
//                b0 <- b0 - 1
//            
//        let mutable extended = if (a1 < destCount) && (b1 < sourceCount) then 1 else 0
//
//        let aCount = a1 - a0
//        let bCount = b1 - b0
//
//        let a_shared = indices_shared.Ptr(0)
//        let b_shared = indices_shared.Ptr(aCount)
//
//        deviceMemToMemLoop (bCount + extended) (b_global + b0) tid b_shared true
//
//        let diag = min (VT * tid) (aCount + bCount - loadPrecedingB)
//        
//        //let countingItr = mpCountingItr + a0
//        //let sharedCountingItr = __local__.Array<int>(itrSize) |> __array_to_ptr
//        //counting_iterator countingItr_global a0 sharedCountingItr
//        //__syncthreads()
//        let mp = mergePath (countingItr_global + a0) aCount (b_shared + loadPrecedingB) (bCount - loadPrecedingB) diag
////        let a = __local__.Array<int>(1) |> __array_to_ptr
////        a.[0] <- a0
////        let b = b_shared + loadPrecedingB
////        let bCount2 = bCount - loadPrecedingB
////        let mp = 
////            let mutable begin' = max 0 (diag - bCount2)
////            let mutable end' = min diag aCount
////
////            while begin' < end' do
////                let mid = (begin' + end') >>> 1
////                let aKey = a.[0] + mid
////                let bKey = b.[diag - 1 - mid]
////
////                let pred = aKey < bKey
////                        
////                if pred then 
////                    begin' <- mid + 1
////                else
////                   end' <- mid
////            begin'
//
//        let a0tid = a0 + mp
//        let b0tid = diag - mp + loadPrecedingB
//
//        if extended <> 0 then
//            deviceSerialLoadBalanceSearchFalse b_shared a0tid a1 (b0 - 1) b0tid bCount (a_shared - a0)
//        else
//            deviceSerialLoadBalanceSearchTrue b_shared a0tid a1 (b0 - 1) b0tid bCount (a_shared - a0)
//        __syncthreads()
//
//        int4(a0, a1, b0, b1)
//            @>