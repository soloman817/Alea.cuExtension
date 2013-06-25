﻿module Alea.CUDA.Extension.MGPU.CTAMerge

open Microsoft.FSharp.Quotations
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Static
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.Intrinsics
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTASearch
//open Alea.CUDA.Extension.MGPU.SortNetwork

[<ReflectedDefinition>]
let doRangeCheck = 1
[<ReflectedDefinition>]
let dontRangeCheck = 0

[<ReflectedDefinition>]
let doesHaveValues = 1
[<ReflectedDefinition>]
let doesntHaveValues = 0


// SerialMerge
let serialMerge (VT:int) (rangeCheck:int) (comp:IComp<'T>) =
    let comp = comp.Device
    <@ fun (keys_shared:RWPtr<'T>) (aBegin:int) (aEnd:int) (bBegin:int) (bEnd:int) (results:RWPtr<'T>) (indices:RWPtr<int>) ->
        let comp = %comp

        let mutable aKey = keys_shared.[aBegin]
        let mutable bKey = keys_shared.[bBegin]
        let mutable aBegin = aBegin
        let mutable bBegin = bBegin

        for i = 0 to VT - 1 do
            let mutable p = 0
            if rangeCheck = 1 then
                p <- if (bBegin >= bEnd) || ((aBegin < aEnd) && ((comp bKey aKey) = 1)) then 1 else 0
            else
                p <- if (comp bKey aKey) = 0 then 1 else 0

            results.[i] <- if p = 1 then aKey else bKey
            indices.[i] <- if p = 1 then aBegin else bBegin

            if p = 1 then
                aBegin <- aBegin + 1
                aKey <- keys_shared.[aBegin]
            else
                bBegin <- bBegin + 1
                bKey <- keys_shared.[bBegin]
        __syncthreads() @>

////////////////////////////////////////////////////////////////////////////////
// FindMergeFrame and FindMergesortInterval help mergesort (both CTA and global 
// merge pass levels) locate lists within the single source array.

// Returns (offset of a, offset of b, length of list).
type FindMergesortFrame() =    
    member fmf.Host =
        fun (coop:int) (block:int) (nv:int) ->
            let start = ~~~(coop - 1) &&& block
            let size = nv * (coop >>> 1)
            int3((nv * start), (nv * start + size), (size))
         
    member fmf.Device =
        <@ fun (coop:int) (block:int) (nv:int) ->
            let start = ~~~(coop - 1) &&& block
            let size = nv * (coop >>> 1)
            int3((nv * start), (nv * start + size), (size)) @>

// Returns (a0, a1, b0, b1) into mergesort input lists between mp0 and mp1.
type FindMergesortInterval() =
    member fmi.Host =
        fun (frame:int3) (coop:int) (block:int) (nv:int) (count:int) (mp0:int) (mp1:int) ->
            let diag = nv * block - frame.x
            let a0 = frame.x + mp0
            let mutable a1 = min count (frame.x + mp1)
            let b0 = min count (frame.y + diag - mp0)
            let mutable b1 = min count (frame.y + diag + nv - mp1)
            if (coop - 1) = ((coop - 1) &&& block) then
                a1 <- min count (frame.x + frame.z)
                b1 <- min count (frame.y + frame.z)
            int4(a0,a1,b0,b1)
            

    member fmi.Device =
        <@ fun (frame:int3) (coop:int) (block:int) (nv:int) (count:int) (mp0:int) (mp1:int) ->
            let diag = nv * block - frame.x
            let a0 = frame.x + mp0
            let mutable a1 = min count (frame.x + mp1)
            let b0 = min count (frame.y + diag - mp0)
            let mutable b1 = min count (frame.y + diag + nv - mp1)
            if (coop - 1) = ((coop - 1) &&& block) then
                a1 <- min count (frame.x + frame.z)
                b1 <- min count (frame.y + frame.z)
            int4(a0,a1,b0,b1) @>

let findMergesortFrame = new FindMergesortFrame()
let findMergesortInterval = new FindMergesortInterval()

// ComputeMergeRange
type ComputeMergeRange() =
    member cmr.Host =
        fun (aCount:int) (bCount:int) (block:int) (coop:int) (NV:int) (mp_global:RWPtr<int>) ->
            let mp0 = mp_global.[block]
            let mp1 = mp_global.[block + 1]
            let gid = NV * block

            let mutable range = int4(0,0,0,0)
            if coop <> 0 then
                let frame = findMergesortFrame.Host coop block NV
                range <- findMergesortInterval.Host frame coop block NV aCount mp0 mp1
            else
                range.x <- mp0
                range.y <- mp1
                range.z <- gid - range.x
                range.w <- min (aCount + bCount) (gid + NV) - range.y
            range

    member cmr.Device =
        let findMergesortFrame = findMergesortFrame.Device
        let findMergesortInterval = findMergesortInterval.Device
        <@ fun (aCount:int) (bCount:int) (block:int) (coop:int) (NV:int) (mp_global:RWPtr<int>) ->
            let findMergesortFrame = %findMergesortFrame
            let findMergesortInterval = %findMergesortInterval

            let mp0 = mp_global.[block]
            let mp1 = mp_global.[block + 1]
            let gid = NV * block

            let mutable range = int4(0,0,0,0)
            if coop <> 0 then
                let frame = findMergesortFrame coop block NV
                range <- findMergesortInterval frame coop block NV aCount mp0 mp1
            else
                range.x <- mp0
                range.y <- mp1
                range.z <- gid - range.x
                range.w <- min (aCount + bCount) (gid + NV) - range.y
            range @>

let computeMergeRange = new ComputeMergeRange()


// CTA mergesort support
let ctaBlocksortPass (NT:int) (VT:int) (comp:IComp<'TC>) =
    let mergePath = (mergeSearch MgpuBoundsLower comp).DMergePath
    let serialMerge = serialMerge VT doRangeCheck comp
    let comp = comp.Device
    <@ fun (keys_shared:RWPtr<'T>) (tid:int) (count:int) (coop:int) (keys:RWPtr<'T>) (indices:RWPtr<int>) ->
        let comp = %comp
        let mergePath = %mergePath
        let serialMerge = %serialMerge

        let list = ~~~(coop - 1) &&& tid
        let diag = min count (VT * ((coop - 1) &&& tid))
        let start = VT * list
        let a0 = min count start
        let b0 = min count (start + VT * (coop / 2))
        let b1 = min count (start + VT * coop)
        let p = mergePath (keys_shared + a0) (b0 - a0) (keys_shared + b0) (b1 - b0) diag
        serialMerge keys_shared (a0 + p) b0 (b0 + diag - p) b1 keys indices @>

let ctaBlocksortLoop (NT:int) (VT:int) (hasValues:int) (comp:IComp<'TC>) =
    let ctaBlocksortPass = ctaBlocksortPass NT VT comp
    let deviceThreadToShared = deviceThreadToShared VT
    let deviceGather = deviceGather NT VT

    <@ fun (threadValues:RWPtr<'TV>) (keys_shared:RWPtr<'TK>) (values_shared:RWPtr<'TV>) (tid:int) (count:int) ->
        let ctaBlocksortPass = %ctaBlocksortPass
        let deviceThreadToShared = %deviceThreadToShared
        let deviceGather = %deviceGather

        let mutable coop = 2
        while coop <= NT do
            let indices = __local__<int>(VT).Ptr(0)
            let keys = __local__<'TK>(VT).Ptr(0)
            ctaBlocksortPass keys_shared tid count coop keys indices

            if hasValues = 1 then
                deviceThreadToShared threadValues tid values_shared doSync
                deviceGather (NT * VT) values_shared indices tid threadValues doSync

            deviceThreadToShared keys tid keys_shared doSync
            coop <- coop * 2 @>

////////////////////////////////////////////////////////////////////////////////
// CTAMergesort
// Caller provides the keys in shared memory. This functions sorts the first
// count elements.
let ctaMergesort (NT:int) (VT:int) (hasValues:int) (comp:IComp<'TC>) =
    let deviceThreadToShared = deviceThreadToShared VT
    let ctaBlocksortLoop = ctaBlocksortLoop NT VT hasValues comp
    
    <@ fun (threadKeys:DevicePtr<'TK>) (threadValues:DevicePtr<'TV>) (keys_shared:RWPtr<'TK>) (values_shared:RWPtr<'TV>) (count:int) (tid:int) (compOp:CompOpType) ->
        let deviceThreadToShared = %deviceThreadToShared
        let ctaBlocksortLoop = %ctaBlocksortLoop
        if (VT * tid) > count then
            oddEvenTransposeSort<'TV> threadsKeys threadValues compOp
        deviceThreadToShared threadKeys tid keys_shared

        ctaBlocksortLoop threadValues keys_shared values_shared tid count compOp @>


let deviceMergeKeysIndices (NT:int) (VT:int) (comp:IComp<'T>) =
    let deviceLoad2ToShared = deviceLoad2ToSharedA NT VT VT
    let mergePath = (mergeSearch MgpuBoundsLower comp).DMergePath
    let serialMerge = serialMerge VT 1 comp

    <@ fun (a_global:RWPtr<'T1>) (b_global:RWPtr<'T2>) (range:int4) (tid:int) (keys_shared:RWPtr<'T>) (results:RWPtr<'T>) (indices:RWPtr<int>) ->
        let deviceLoad2ToShared = %deviceLoad2ToShared
        let mergePath = %mergePath
        let serialMerge = %serialMerge

        let a0 = range.x
        let a1 = range.y
        let b0 = range.z
        let b1 = range.w
        let aCount = a1 - a0
        let bCount = b1 - b0

        deviceLoad2ToShared (a_global + a0) (b_global + b0) bCount tid keys_shared doSync

        let diag = VT * tid
        let mp = mergePath keys_shared aCount (keys_shared + aCount) bCount diag // comp

        let a0tid = mp
        let a1tid = aCount
        let b0tid = aCount + diag - mp
        let b1tid = aCount + bCount

        serialMerge keys_shared a0tid a1tid b0tid b1tid results indices // comp
        @>


let deviceMerge (NT:int) (VT:int) (hasValues:int) (comp:IComp<'TC>) =
    let deviceMergeKeysIndices = deviceMergeKeysIndices NT VT comp
    let deviceThreadToShared = deviceThreadToShared VT
    let deviceSharedToGlobal = deviceSharedToGlobal NT VT
    let deviceTransferMergeValues = deviceTransferMergeValuesA NT VT

    <@ fun (aKeys_global:RWPtr<'K1>) (aVals_global:RWPtr<'V1>) (bKeys_global:RWPtr<'K2>) (bVals_global:RWPtr<'V2>) (tid:int) (block:int) (range:int4) (keys_shared:RWPtr<'TK>) (indices_shared:RWPtr<int>) (keys_global:RWPtr<'K3>) (vals_global:RWPtr<'V3>) ->
        let deviceMergeKeysIndices = %deviceMergeKeysIndices
        let deviceThreadToShared = %deviceThreadToShared
        let deviceSharedToGlobal = %deviceSharedToGlobal
        let deviceTransferMergeValues = %deviceTransferMergeValues

        let results = __local__<'TK>(VT).Ptr(0)
        let indices = __local__<int>(VT).Ptr(0)
        deviceMergeKeysIndices aKeys_global bKeys_global range tid keys_shared results indices // comp
        
        deviceThreadToShared results tid keys_shared

        let aCount = range.y - range.x
        let bCount = range.w - range.z
        deviceSharedToGlobal (aCount + bCount) keys_shared tid (keys_global + NT * NT * block)

        if hasValues = 1 then
            deviceThreadToShared indices tid indices_shared

            deviceTransferMergeValues (aCount + bCount) (aVals_global + range.x) (bVals_global + range.z) aCount indices_shared tid (vals_global + NT * VT * block)
        @>