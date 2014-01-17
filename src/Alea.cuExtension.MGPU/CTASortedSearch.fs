[<AutoOpen>]
module Alea.cuExtension.MGPU.CTASortedSearch

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.cuExtension
//open Alea.cuExtension.Util
open Alea.cuExtension.MGPU
//open Alea.cuExtension.MGPU.QuotationUtil
open Alea.cuExtension.MGPU.DeviceUtil
open Alea.cuExtension.MGPU.LoadStore
open Alea.cuExtension.MGPU.CTAScan
open Alea.cuExtension.MGPU.CTASearch
open Alea.cuExtension.MGPU.CTAMerge

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


let deviceSerialSearch (VT:int) (bounds:int) (rangeCheck:bool) (indexA:bool) (matchA:bool) (indexB:bool) (matchB:bool) (compOp:IComp<'TK>) =
    
    let compId = compOp.Identity
    let comp = compOp.Device
    <@ fun (keys_shared:deviceptr<'TK>) (aBegin:int) (aEnd:int) (bBegin:int) (bEnd:int) (aOffset:int) (bOffset:int) (indices:deviceptr<int>) ->
        let comp = %comp
        //let compId = %compId

        let flagA = if indexA then 0x80000000 else 1
        let flagB = if indexB then 0x80000000 else 1

        //let keys_shared = keys___shared__.Array<int>()

        let mutable aKey = keys_shared.[aBegin]
        let mutable bKey = keys_shared.[bBegin]
        let mutable aPrev = compId
        let mutable bPrev = compId

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
                        let inRange = (not rangeCheck) || ((bBegin < bEnd) && (aBegin < aEnd))
                        match' <- inRange && not (comp bKey aKey)
                    else
                        let inRange = (not rangeCheck) || ((bBegin < bEnd) && (aBegin > 0))
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
    


let ctaSortedSearch (NT:int) (VT:int) (bounds:int) (indexA:bool) (matchA:bool) (indexB:bool) (matchB:bool) (compOp:IComp<'TK>) =
    let NV = NT * VT
    let mergePath = (mergePath bounds compOp).DMergePath
    let deviceSerialSearch1 = deviceSerialSearch VT bounds false indexA matchA indexB matchB compOp
    let deviceSerialSearch2 = deviceSerialSearch VT bounds true indexA matchA indexB matchB compOp

    <@ fun (keys_shared:deviceptr<'TK>) (aStart:int) (aCount:int) (aEnd:int) (a0:int) (bStart:int) (bCount:int) (bEnd:int) (b0:int) (extended:bool) (tid:int) (indices_shared:deviceptr<int>) ->
        let mergePath = %mergePath
        let deviceSerialSearch1 = %deviceSerialSearch1
        let deviceSerialSearch2 = %deviceSerialSearch2
                        
        let diag = VT * tid
        let mp = mergePath (keys_shared + aStart) aCount (keys_shared + bStart) bCount diag
        let mutable a0tid = mp
        let mutable b0tid = diag - mp

        let indices = __local__.Array<int>(VT) |> __array_to_ptr
        let mutable results = int3(0,0,0)
        if extended then
            results <- deviceSerialSearch1 keys_shared (a0tid + aStart) aEnd (b0tid + bStart) bEnd (a0 - aStart) (b0 - bStart) indices
        else
            results <- deviceSerialSearch2 keys_shared (a0tid + aStart) aEnd (b0tid + bStart) bEnd (a0 - aStart) (b0 - bStart) indices
        __syncthreads()

        let mutable decisions = results.x
        b0tid <- b0tid + aCount
        for i = 0 to VT - 1 do
            if ((1 <<< i) &&& decisions) > 0 then
                if (indexA || matchA) then 
                    indices_shared.[a0tid] <- indices.[i]
                    a0tid <- a0tid + 1
            else
                if (indexB || matchB) then
                    indices_shared.[b0tid] <- indices.[i]
                    b0tid <- b0tid + 1
        __syncthreads()

        int2(results.y, results.z)
     @>
                    