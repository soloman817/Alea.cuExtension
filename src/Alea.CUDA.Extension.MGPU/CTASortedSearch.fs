module Alea.CUDA.Extension.MGPU.CTASortedSearch
// NOT IMPLEMENTED YET
open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTAScan
open Alea.CUDA.Extension.MGPU.CTASearch
open Alea.CUDA.Extension.MGPU.CTAMerge

//
//type Plan<'T> =
//    {
//        NT : int
//        VT : int
//        Bounds : int
//        RangeCheck : bool
//        IndexA : bool
//        MatchA : bool
//        IndexB : bool
//        MatchB : bool
//        CompOp : IComp<'T>
//    }


let deviceSerialSearch (VT:int) (bounds:int) (rangeCheck:bool) (indexA:bool) (matchA:bool) (indexB:bool) (matchB:bool) (compOp:IComp<'TC>) =
    

    let comp = compOp.Device
    <@ fun (keys_shared:RWPtr<int>) (aBegin:int) (aEnd:int) (bBegin:int) (bEnd:int) (aOffset:int) (bOffset:int) (indices:DevicePtr<int>) ->
        let comp = %comp

        let flagA = if indexA then 0x80000000 else 1
        let flagB = if indexB then 0x80000000 else 1

        let mutable aKey = keys_shared.[aBegin]
        let mutable bKey = keys_shared.[bBegin]
        let mutable aPrev = 0
        let mutable bPrev = 0

        let mutable aBegin = aBegin
        let mutable bBegin = bBegin

        if aBegin > 0 then aPrev <- keys_shared.[aBegin - 1]
        if bBegin > 0 then bPrev <- keys_shared.[bBegin - 1]
        let mutable decisions = 0
        let mutable matchCountA = 0
        let mutable matchCountB = 0

        for i = 0 to VT - 1 do
            let mutable p = false
            if rangeCheck && (aBegin >= aEnd) then p <- false
            elif rangeCheck && (bBegin >= bEnd) then p <- true
            else p <- if bounds = MgpuBoundsUpper then (comp aKey bKey) else not (comp bKey aKey)

            if p then
                let mutable match' = false
                if matchA then
                    if bounds = MgpuBoundsUpper then
                        let inRange = (not rangeCheck) || (bBegin > aEnd)
                        match' <- inRange && not (comp bPrev aKey)
                else
                    let inRange = (not rangeCheck) || (bBegin < bEnd)
                    match' <- inRange && not (comp aKey bKey)
            
                let mutable index = 0
                if indexA then index <- bOffset + bBegin
                if match' then index <- index ||| flagA
                if (indexA || matchA) then indices.[i] <- index
                matchCountA <- matchCountA + (if match' then 1 else 0)

                decisions <- decisions ||| (1 <<< i)
                aPrev <- aKey
                aBegin <- aBegin + 1
                aKey <- keys_shared.[aBegin]
            else
                let mutable match' = false
                if matchB then
                    if MgpuBoundsUpper = bounds then
                        let inRange = (not rangeCheck) || ((aBegin < bEnd) && (aBegin > 0))
                        match' <- inRange && not (comp aPrev bKey)
                    else
                        let inRange = (not rangeCheck) || ((aBegin < bEnd) && (aBegin > 0))
                        match' <- inRange && not (comp aPrev bKey)

                let mutable index = 0
                if indexB then index <- aOffset + aBegin
                if match' then index <- index ||| flagB
                if (indexB || matchB) then indices.[i] <- index
                matchCountB <- matchCountB + (if match' then 1 else 0)
                
                bPrev <- bKey
                bBegin <- bBegin + 1
                bKey <- keys_shared.[bBegin]

        int3(decisions, matchCountA, matchCountB)

        @>
    


let ctaSortedSearch (NT:int) (VT:int) (bounds:int) (indexA:bool) (matchA:bool) (indexB:bool) (matchB:bool) (compOp:IComp<'TC>) =
    
    let NV = NT * VT
    
    let mergePath = (mergeSearch bounds (comp CompTypeLess 0)).DMergePath
    let deviceSerialSearch1 = deviceSerialSearch VT bounds false indexA matchA indexB matchB compOp
    let deviceSerialSearch2 = deviceSerialSearch VT bounds true indexA matchA indexB matchB compOp

    let sharedSize = NV

    <@ fun (keys_shared:RWPtr<int>) (aStart:int) (aCount:int) (aEnd:int) (a0:int) (bStart:int) (bCount:int) (bEnd:int) (b0:int) (extended:bool) (tid:int) (indices_shared:RWPtr<int>) ->
        let mergePath = %mergePath
        let deviceSerialSearch1 = %deviceSerialSearch1
        let deviceSerialSearch2 = %deviceSerialSearch2

        let shared = __shared__<int>(sharedSize).Ptr(0)
        
        let diag = VT * tid
        let ksas = DevicePtr(keys_shared.Handle64 + int64(aStart))
        let ksbs = DevicePtr(keys_shared.Handle64 + int64(bStart))
        let mp = mergePath ksas aCount ksbs bCount diag
        let mutable a0tid = mp
        let mutable b0tid = diag - mp

        let indices = __local__<int>(VT).Ptr(0)
        let mutable results = int3(0,0,0)
        if extended then
            results <- deviceSerialSearch1 keys_shared (a0tid + aStart) aEnd (b0tid + bStart) bEnd (a0 - aStart) (b0 - bStart) (DevicePtr(indices.Handle64))
        else
            results <- deviceSerialSearch2 keys_shared (a0tid + aStart) aEnd (b0tid + bStart) bEnd (a0 - aStart) (b0 - bStart) (DevicePtr(indices.Handle64))
        __syncthreads()

        let mutable decisions = results.x
        b0tid <- b0tid + aCount
        for i = 0 to VT - 1 do
            if ((1 <<< i) &&& decisions) <> 0 then
                if (indexA || matchA) then 
                    indices_shared.[a0tid] <- indices.[i]
                    a0tid <- a0tid + 1
                if (indexB || matchB) then
                    indices_shared.[b0tid] <- indices.[i]
                    b0tid <- b0tid + 1
        __syncthreads()

        int2(results.y, results.z)
     @>
                    