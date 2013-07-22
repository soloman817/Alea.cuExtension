module Alea.CUDA.Extension.MGPU.CTALoadBalance
// PARTIAL IMPLEMENTATION
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
open Alea.CUDA.Extension.MGPU.CTAMerge




////////////////////////////////////////////////////////////////////////////////
// DeviceLoadBalancingSearch
// Upper Bound search from A (needles) into B (haystack). The A values are 
// natural numbers from aBegin to aEnd. bFirst is the index of the B value at
// bBegin in shared memory.

//template<int VT, bool RangeCheck>
let deviceSerialLoadBalanceSearch (VT:int) (rangeCheck:int) =
    <@ fun (b_shared:RWPtr<int>) (aBegin:int) (aEnd:int) (bFirst:int) (bBegin:int) (bEnd:int) (a_shared:RWPtr<int>) -> 
        let mutable bKey = b_shared.[bBegin]
        let mutable aBegin = aBegin
        let mutable bBegin = bBegin

        for i = 0 to VT - 1 do
            let mutable p = 0
            if rangeCheck = 1 then
                p <- if (aBegin < aEnd) && ((bBegin >= bEnd) || (aBegin < bKey)) then 1 else 0
            else
                p <- if aBegin < bKey then 1 else 0

            if p = 1 then
                // Advance A (the needle).
                a_shared.[aBegin] <- bFirst + bBegin
                aBegin <- aBegin + 1
            else
                // Advance B (the haystack).
                bBegin <- bBegin + 1
                bKey <- b_shared.[bBegin] @>


//// CTALoadBalance
let ctaLoadBalance (NT:int) (VT:int) =
    let computeMergeRange = computeMergeRange.Device
    let mergePath = (mergeSearch MgpuBoundsUpper (comp CompTypeLess 0)).DMergePath
    let deviceMemToMemLoop = deviceMemToMemLoop NT
    let deviceSerialLoadBalanceSearchFalse = deviceSerialLoadBalanceSearch VT 0
    let deviceSerialLoadBalanceSearchTrue = deviceSerialLoadBalanceSearch VT 1

    <@ fun (destCount:int) (b_global:RWPtr<int>) (sourceCount:int) (block:int) (tid:int) (mp_global:DevicePtr<int>) (indices_shared:RWPtr<int>) (loadPrecedingB:int) ->
        let computeMergeRange = %computeMergeRange
        let mergePath = %mergePath
        let deviceMemToMemLoop = %deviceMemToMemLoop
        let deviceSerialLoadBalanceSearchFalse = %deviceSerialLoadBalanceSearchFalse
        let deviceSerialLoadBalanceSearchTrue = %deviceSerialLoadBalanceSearchTrue

        let mutable loadPrecedingB = loadPrecedingB

        let range = computeMergeRange destCount sourceCount block 0 (NT * VT) mp_global
        
        let a0 = range.x
        let a1 = range.y
        let mutable b0 = range.z
        let b1 = range.w

        let mutable loadPrecedingB = loadPrecedingB
        if loadPrecedingB = 1 then
            if b0 = 0 then
                loadPrecedingB <- 0
            else
                b0 <- b0 - 1
            
        let mutable extended = if (a1 < destCount) && (b1 < sourceCount) then 1 else 0

        let mutable aCount = a1 - a0
        let mutable bCount = b1 - b0

        let a_shared = indices_shared
        let b_shared = indices_shared + aCount

        deviceMemToMemLoop (bCount + extended) (b_global + b0) tid b_shared true

        let diag = min (VT * tid) (aCount + bCount - loadPrecedingB)
        let newBshared = DevicePtr(b_shared.Handle64 + int64(loadPrecedingB))

        let mp = mergePath (DevicePtr(int64 a0)) aCount newBshared (bCount - loadPrecedingB) diag

        let a0tid = a0 + mp
        let b0tid = diag - mp + loadPrecedingB

        if extended = 1 then
            deviceSerialLoadBalanceSearchFalse b_shared a0tid a1 (b0 - 1) b0tid bCount (a_shared - a0)
        else
            deviceSerialLoadBalanceSearchTrue b_shared a0tid a1 (b0 - 1) b0tid bCount (a_shared - a0)
        __syncthreads()

        int4(a0, a1, b0, b1)
            @>