module Alea.CUDA.Extension.MGPU.Mergesort

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.Intrinsics
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTAScan
open Alea.CUDA.Extension.MGPU.CTASearch
open Alea.CUDA.Extension.MGPU.CTAMerge
open Alea.CUDA.Extension.MGPU.Merge




let kernelBlocksort (plan:Plan) (hasValues:int) (compOp:IComp<'TV>) =
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    let hasValues = if hasValues = 1 then true else false

    let sharedSize = (NT * (VT + 1))
    let comp = compOp.Device
    
    let deviceGlobalToShared = deviceGlobalToShared NT VT
    let deviceSharedToThread = deviceSharedToThread VT
    let deviceSharedToGlobal = deviceSharedToGlobal NT VT
    let deviceThreadToShared = deviceThreadToShared VT
    let ctaMergesort = ctaMergesort NT VT hasValues compOp
                                                    
    <@ fun  (keysSource_global  :DevicePtr<'TV>) 
            (valsSource_global  :DevicePtr<'TV>) 
            (count              :int) 
            (keysDest_global    :DevicePtr<'TV>) 
            (valsDest_global    :DevicePtr<'TV>) ->

        let comp = %comp
        let deviceGlobalToShared = %deviceGlobalToShared
        let deviceSharedToThread = %deviceSharedToThread
        let deviceSharedToGlobal = %deviceSharedToGlobal
        let deviceThreadToShared = %deviceThreadToShared
        let ctaMergesort = %ctaMergesort

        let shared = __shared__<'TV>(sharedSize).Ptr(0)
        let sharedKeys = shared
        let sharedValues = shared

        let tid = threadIdx.x
        let block = blockIdx.x
        let gid = NV * block
        let count2 = min NV (count - gid)            

        let threadValues = __local__<'TV>(VT).Ptr(0) 
        if hasValues then
            deviceGlobalToShared count2 (valsSource_global + gid) tid sharedValues true
            deviceSharedToThread sharedValues tid threadValues true
        
        let threadKeys = __local__<'TV>(VT).Ptr(0)
        deviceGlobalToShared count2 (keysSource_global + gid) tid sharedKeys true
        deviceSharedToThread sharedKeys tid threadKeys true

        let first = VT * tid
        if ((first + VT) > count2) && (first < count2) then
            let mutable maxKey = threadKeys.[0]    
            for i = 1 to VT - 1 do
                if (first + i) < count2 then
                    maxKey <- if comp maxKey threadKeys.[i] then threadKeys.[i] else maxKey
            for i = 0 to VT - 1 do
                if (first + i) >= count2 then threadKeys.[i] <- maxKey

        ctaMergesort threadKeys threadValues sharedKeys sharedValues count2 tid
        deviceSharedToGlobal count2 sharedKeys tid (keysDest_global + gid) true

        if hasValues then
            deviceThreadToShared threadValues tid sharedValues true
            deviceSharedToGlobal count2 sharedValues tid (valsDest_global + gid) true
        @>


type IBlocksort<'TV> =
    {
        Action : ActionHint -> DevicePtr<'TV> -> DevicePtr<'TV> -> DevicePtr<int> -> unit
        NumPartitions : int
    }


let mergesortKeys (compOp:IComp<'TV>) = cuda {
    let plan = { NT = 256; VT = 7 }
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT
    
    let! kernelBlocksort = kernelBlocksort plan 0 compOp |> defineKernelFuncWithName "kbs"
    let! mpp = Search.mergePathPartitions MgpuBoundsLower compOp    
    let! kernelMerge = Merge.pKernelMergesort {NT = 256; VT = 7} compOp
    

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelBlocksort = kernelBlocksort.Apply m
        let mpp = mpp.Apply m
        let kernelMerge = kernelMerge.Apply m

        fun (count:int) ->
            let numBlocks = divup count NV
            let numPasses = findLog2 numBlocks true
            let lp = LaunchParam(numBlocks, NT)

            let action (hint:ActionHint) (source:DevicePtr<'TV>) (dest:DevicePtr<'TV>) (parts:DevicePtr<int>) =
                fun () ->
                    let lp = lp |> hint.ModifyLaunchParam
                    kernelBlocksort.Launch lp source (DevicePtr<'TV>(0n)) count (if (1 &&& numPasses) <> 0 then dest else source) (DevicePtr<'TV>(0n))
                    
                    if (1 &&& numPasses) <> 0 then
                        swap source dest
                    for pass = 0 to numPasses - 1 do
                        let coop = 2 <<< pass
                        let mpp = mpp count 0 NV coop
                        let partitions = mpp.Action hint source source parts
                        let kernelMerge = kernelMerge count coop                        
                        let merged = kernelMerge.Action hint source parts dest
                        swap dest source
                |> worker.Eval
            { Action = action; NumPartitions = numBlocks + 1 } ) }



type IMergesortPairs<'TV> =
    {
        Action : ActionHint -> DevicePtr<'TV> -> DevicePtr<'TV> -> DevicePtr<'TV> -> DevicePtr<'TV> -> DevicePtr<int> -> unit
        NumPartitions : int
    }

//////////////////////////////////////////////////////////////////////////////////
//// MergesortPairs
//
let mergesortPairs (compOp:IComp<'TV>) = cuda {
    let plan = { NT = 256; VT = 11 }
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    let! kernelBlocksort = kernelBlocksort plan 1 compOp |> defineKernelFuncWithName "kbs"
    let! mpp = Search.mergePathPartitions MgpuBoundsLower compOp
    let! kernelMerge = kernelMerge plan 1 1 compOp |> defineKernelFuncWithName "km"

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelBlocksort = kernelBlocksort.Apply m
        let mpp = mpp.Apply m
        let kernelMerge = kernelMerge.Apply m

        fun (count:int) ->
            let numBlocks = divup count NV
            let numPasses = findLog2 numBlocks true
            let lp = LaunchParam(numBlocks, NT)
            let action (hint:ActionHint) (keysSource:DevicePtr<'TV>) (valsSource:DevicePtr<'TV>) (keysDest:DevicePtr<'TV>) (valsDest:DevicePtr<'TV>) (parts:DevicePtr<int>) =
                fun () ->
                    let lp = lp |> hint.ModifyLaunchParam
                    kernelBlocksort.Launch lp keysSource valsSource count (if (1 &&& numPasses) <> 0 then keysDest else keysSource) (if (1 &&& numPasses) <> 0 then valsDest else valsSource)

                    if (1 &&& numPasses) <> 0 then
                        swap keysSource keysDest
                        swap valsSource valsDest

                    for pass = 0 to numPasses - 1 do
                        let coop = 2 <<< pass
                        let mpp = mpp count 0 NV coop
                        let partitions = mpp.Action hint keysSource keysSource parts
                        kernelMerge.Launch lp keysSource valsSource count keysSource valsSource 0 parts coop keysDest valsDest
                        swap keysDest keysSource
                        swap valsDest valsSource

                |> worker.Eval
            { Action = action; NumPartitions = numBlocks + 1 } ) }



type IMergesortIndices<'TV> =
    {
        Action : ActionHint -> DevicePtr<'TV> -> DevicePtr<'TV> -> DevicePtr<'TV> -> DevicePtr<'TV> -> DevicePtr<'TV> -> DevicePtr<int> -> unit
        NumPartitions : int
    }

//////////////////////////////////////////////////////////////////////////////////
//// MergesortIndices
//
let mergesortIndices (compOp:IComp<int>) = cuda {
    let plan = { NT = 256; VT = 11 }
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    let! kernelBlocksort = kernelBlocksort plan 1 compOp |> defineKernelFuncWithName "kbs"
    let! mpp = Search.mergePathPartitions MgpuBoundsLower compOp
    let! kernelMerge = kernelMerge plan 1 1 compOp |> defineKernelFuncWithName "km"

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelBlocksort = kernelBlocksort.Apply m
        let mpp = mpp.Apply m
        let kernelMerge = kernelMerge.Apply m

        fun (count:int) ->
            let numBlocks = divup count NV
            let numPasses = findLog2 numBlocks true
            let lp = LaunchParam(numBlocks, NT)
            let action (hint:ActionHint) (keysSource:DevicePtr<int>) (countingItr:DevicePtr<int>) (valsSource:DevicePtr<int>) (keysDest:DevicePtr<int>) (valsDest:DevicePtr<int>) (parts:DevicePtr<int>) =
                fun () ->
                    let lp = lp |> hint.ModifyLaunchParam
                    kernelBlocksort.Launch lp keysSource countingItr count (if (1 &&& numPasses) <> 0 then keysDest else keysSource) (if (1 &&& numPasses) <> 0 then valsDest else valsSource)

                    if (1 &&& numPasses) <> 0 then
                        swap keysSource keysDest
                        swap valsSource valsDest

                    for pass = 0 to numPasses - 1 do
                        let coop = 2 <<< pass
                        let mpp = mpp count 0 NV coop
                        let partitions = mpp.Action hint keysSource keysSource parts
                        kernelMerge.Launch lp keysSource valsSource count keysSource valsSource 0 parts coop keysDest valsDest
                        swap keysDest keysSource
                        swap valsDest valsSource

                |> worker.Eval
            { Action = action; NumPartitions = numBlocks + 1 } ) }