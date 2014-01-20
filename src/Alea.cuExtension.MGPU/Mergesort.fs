module Alea.cuExtension.MGPU.Mergesort
//
//open System.Runtime.InteropServices
//open Microsoft.FSharp.Collections
//open Alea.CUDA
//open Alea.cuExtension
////open Alea.cuExtension.Util
//open Alea.cuExtension.MGPU
//open Alea.cuExtension.MGPU.Intrinsics
////open Alea.cuExtension.MGPU.QuotationUtil
//open Alea.cuExtension.MGPU.DeviceUtil
//open Alea.cuExtension.MGPU.LoadStore
//open Alea.cuExtension.MGPU.CTAScan
//open Alea.cuExtension.MGPU.CTASearch
//open Alea.cuExtension.MGPU.CTAMerge
//open Alea.cuExtension.MGPU.Merge
//
//
//
//
//let kernelBlocksort (plan:Plan) (hasValues:int) (compOp:IComp<'TV>) =
//    let NT = plan.NT
//    let VT = plan.VT
//    let NV = NT * VT
//
//    let hasValues = if hasValues = 1 then true else false
//
//    let sharedSize = (NT * (VT + 1))
//    let comp = compOp.Device
//    
//    let deviceGlobalToShared = deviceGlobalToShared NT VT
//    let deviceSharedToThread = deviceSharedToThread VT
//    let deviceSharedToGlobal = deviceSharedToGlobal NT VT
//    let deviceThreadToShared = deviceThreadToShared VT
//    let ctaMergesort = ctaMergesort NT VT hasValues compOp
//                                                    
//    <@ fun  (keysSource_global  :deviceptr<'TV>) 
//            (valsSource_global  :deviceptr<'TV>) 
//            (count              :int) 
//            (keysDest_global    :deviceptr<'TV>) 
//            (valsDest_global    :deviceptr<'TV>) ->
//
//        let comp = %comp
//        let deviceGlobalToShared = %deviceGlobalToShared
//        let deviceSharedToThread = %deviceSharedToThread
//        let deviceSharedToGlobal = %deviceSharedToGlobal
//        let deviceThreadToShared = %deviceThreadToShared
//        let ctaMergesort = %ctaMergesort
//
//        let shared = __shared__<'TV>(sharedSize) |> __array_to_ptr
//        let sharedKeys = shared
//        let sharedValues = shared
//
//        let tid = threadIdx.x
//        let block = blockIdx.x
//        let gid = NV * block
//        let count2 = min NV (count - gid)            
//
//        let threadValues = __local__.Array<'TV>(VT) |> __array_to_ptr 
//        if hasValues then
//            deviceGlobalToShared count2 (valsSource_global + gid) tid sharedValues true
//            deviceSharedToThread sharedValues tid threadValues true
//        
//        let threadKeys = __local__.Array<'TV>(VT) |> __array_to_ptr
//        deviceGlobalToShared count2 (keysSource_global + gid) tid sharedKeys true
//        deviceSharedToThread sharedKeys tid threadKeys true
//
//        let first = VT * tid
//        if ((first + VT) > count2) && (first < count2) then
//            let mutable maxKey = threadKeys.[0]    
//            for i = 1 to VT - 1 do
//                if (first + i) < count2 then
//                    maxKey <- if comp maxKey threadKeys.[i] then threadKeys.[i] else maxKey
//            for i = 0 to VT - 1 do
//                if (first + i) >= count2 then threadKeys.[i] <- maxKey
//
//        ctaMergesort threadKeys threadValues sharedKeys sharedValues count2 tid
//        deviceSharedToGlobal count2 sharedKeys tid (keysDest_global + gid) true
//
//        if hasValues then
//            deviceThreadToShared threadValues tid sharedValues true
//            deviceSharedToGlobal count2 sharedValues tid (valsDest_global + gid) true
//        @>
//
//
//type IBlocksort<'TV> =
//    {
//        Action : ActionHint -> deviceptr<'TV> -> deviceptr<'TV> -> deviceptr<int> -> unit
//        NumPartitions : int
//    }
//
//
//let mergesortKeys (compOp:IComp<'TV>) = cuda {
//    let plan = { NT = 256; VT = 7 }
//    let NT = plan.NT
//    let VT = plan.VT
//    let NV = NT * VT
//    
//    let! kernelBlocksort = kernelBlocksort plan 0 compOp |> Compiler.DefineKernel //"kbs"
//    let! mpp = Search.mergePathPartitions MgpuBoundsLower compOp    
//    let! kernelMerge = Merge.pKernelMergesort plan compOp
//    
//
//    return Entry(fun program ->
//        let worker = program.Worker
//        let kernelBlocksort = kernelBlocksort.Apply m
//        let mpp = mpp.Apply m
//        let kernelMerge = kernelMerge.Apply m
//
//        fun (count:int) ->
//            let numBlocks = divup count NV
//            let numPasses = findLog2 numBlocks true
//            let lp = LaunchParam(numBlocks, NT)
//
//            let run (source:deviceptr<'TV>) (dest:deviceptr<'TV>) (parts:deviceptr<int>) =
//                fun () ->
//                    
//                    //kernelBlocksort.Launch lp source (deviceptr<'TV>(0n)) count (if (1 &&& numPasses) <> 0 then dest else source) (deviceptr<'TV>(0n))
//                    kernelBlocksort.Launch lp source (deviceptr<'TV>(0n)) count dest (deviceptr<'TV>(0n))
//                    
////                    if (1 &&& numPasses) <> 0 then
////                        swap source dest
////                    for pass = 0 to numPasses - 1 do
////                        let coop = 2 <<< pass
////                        let mpp = mpp count 0 NV coop
////                        let partitions = mppsource source parts
////                        let kernelMerge = kernelMerge count coop                        
////                        let merged = kernelMergesource parts dest
////                        swap dest source
//                |> worker.Eval
//            { NumPartitions = numBlocks + 1 } ) }
//
//
//
//type IMergesortPairs<'TV> =
//    {
//        Action : ActionHint -> deviceptr<'TV> -> deviceptr<'TV> -> deviceptr<'TV> -> deviceptr<'TV> -> deviceptr<int> -> unit
//        NumPartitions : int
//    }
//
////////////////////////////////////////////////////////////////////////////////////
////// MergesortPairs
////
//let mergesortPairs (compOp:IComp<'TV>) = cuda {
//    let plan = { NT = 256; VT = 11 }
//    let NT = plan.NT
//    let VT = plan.VT
//    let NV = NT * VT
//
//    let! kernelBlocksort = kernelBlocksort plan 1 compOp |> Compiler.DefineKernel //"kbs"
//    let! mpp = Search.mergePathPartitions MgpuBoundsLower compOp
//    let! kernelMerge = kernelMerge plan 1 1 compOp |> Compiler.DefineKernel //"km"
//
//    return Entry(fun program ->
//        let worker = program.Worker
//        let kernelBlocksort = kernelBlocksort.Apply m
//        let mpp = mpp.Apply m
//        let kernelMerge = kernelMerge.Apply m
//
//        fun (count:int) ->
//            let numBlocks = divup count NV
//            let numPasses = findLog2 numBlocks true
//            let lp = LaunchParam(numBlocks, NT)
//            let run (keysSource:deviceptr<'TV>) (valsSource:deviceptr<'TV>) (keysDest:deviceptr<'TV>) (valsDest:deviceptr<'TV>) (parts:deviceptr<int>) =
//                fun () ->
//                    
//                    kernelBlocksort.Launch lp keysSource valsSource count (if (1 &&& numPasses) <> 0 then keysDest else keysSource) (if (1 &&& numPasses) <> 0 then valsDest else valsSource)
//
//                    if (1 &&& numPasses) <> 0 then
//                        swap keysSource keysDest
//                        swap valsSource valsDest
//
//                    for pass = 0 to numPasses - 1 do
//                        let coop = 2 <<< pass
//                        let mpp = mpp count 0 NV coop
//                        let partitions = mppkeysSource keysSource parts
//                        kernelMerge.Launch lp keysSource valsSource count keysSource valsSource 0 parts coop keysDest valsDest
//                        swap keysDest keysSource
//                        swap valsDest valsSource
//
//                |> worker.Eval
//            { NumPartitions = numBlocks + 1 } ) }
//
//
//
//type IMergesortIndices<'TV> =
//    {
//        Action : ActionHint -> deviceptr<'TV> -> deviceptr<'TV> -> deviceptr<'TV> -> deviceptr<'TV> -> deviceptr<'TV> -> deviceptr<int> -> unit
//        NumPartitions : int
//    }
//
////////////////////////////////////////////////////////////////////////////////////
////// MergesortIndices
////
//let mergesortIndices (compOp:IComp<int>) = cuda {
//    let plan = { NT = 256; VT = 11 }
//    let NT = plan.NT
//    let VT = plan.VT
//    let NV = NT * VT
//
//    let! kernelBlocksort = kernelBlocksort plan 1 compOp |> Compiler.DefineKernel //"kbs"
//    let! mpp = Search.mergePathPartitions MgpuBoundsLower compOp
//    let! kernelMerge = kernelMerge plan 1 1 compOp |> Compiler.DefineKernel //"km"
//
//    return Entry(fun program ->
//        let worker = program.Worker
//        let kernelBlocksort = kernelBlocksort.Apply m
//        let mpp = mpp.Apply m
//        let kernelMerge = kernelMerge.Apply m
//
//        fun (count:int) ->
//            let numBlocks = divup count NV
//            let numPasses = findLog2 numBlocks true
//            let lp = LaunchParam(numBlocks, NT)
//            let run (keysSource:deviceptr<int>) (countingItr:deviceptr<int>) (valsSource:deviceptr<int>) (keysDest:deviceptr<int>) (valsDest:deviceptr<int>) (parts:deviceptr<int>) =
//                fun () ->
//                    
//                    kernelBlocksort.Launch lp keysSource countingItr count (if (1 &&& numPasses) <> 0 then keysDest else keysSource) (if (1 &&& numPasses) <> 0 then valsDest else valsSource)
//
//                    if (1 &&& numPasses) <> 0 then
//                        swap keysSource keysDest
//                        swap valsSource valsDest
//
//                    for pass = 0 to numPasses - 1 do
//                        let coop = 2 <<< pass
//                        let mpp = mpp count 0 NV coop
//                        let partitions = mppkeysSource keysSource parts
//                        kernelMerge.Launch lp keysSource valsSource count keysSource valsSource 0 parts coop keysDest valsDest
//                        swap keysDest keysSource
//                        swap valsDest valsSource
//
//                |> worker.Eval
//            { NumPartitions = numBlocks + 1 } ) }
