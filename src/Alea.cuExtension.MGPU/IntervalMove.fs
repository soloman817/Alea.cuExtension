module Alea.cuExtension.MGPU.IntervalMove
//
//open System.Runtime.InteropServices
//open Microsoft.FSharp.Collections
//open Alea.CUDA
//open Alea.CUDA.Utilities
//open Alea.cuExtension
////open Alea.cuExtension.Util
//open Alea.cuExtension.MGPU
////open Alea.cuExtension.MGPU.QuotationUtil
//open Alea.cuExtension.MGPU.DeviceUtil
//open Alea.cuExtension.MGPU.LoadStore
//open Alea.cuExtension.MGPU.CTALoadBalance
//
//
//
//
////////////////////////////////////////////////////////////////////////////////////
////// KernelIntervalExpand
////
//let kernelIntervalExpand (plan:Plan) =
//    let NT = plan.NT
//    let VT = plan.VT
//    let NV = NT * VT
//
//    let ctaLoadBalance = ctaLoadBalance NT VT
//    let deviceSharedToReg = deviceSharedToReg NT VT
//    let deviceMemToMemLoop = deviceMemToMemLoop NT
//    let deviceGather = deviceGather NT VT
//    let deviceRegToGlobal = deviceRegToGlobal NT VT
//
//    let sharedSize = NT * (VT + 1)
//
//    <@ fun  (destCount          :int) 
//            (indices_global     :deviceptr<int>) 
//            (values_global      :deviceptr<'T>) 
//            (sourceCount        :int)
//            (countingItr_global :deviceptr<int>) 
//            (mp_global          :deviceptr<int>) 
//            (output_global      :deviceptr<'T>) 
//            ->
//        let ctaLoadBalance = %ctaLoadBalance
//        let deviceSharedToReg = %deviceSharedToReg
//        let deviceMemToMemLoop = %deviceMemToMemLoop
//        let deviceGather = %deviceGather
//        let deviceRegToGlobal = %deviceRegToGlobal
//
//        let mutable destCount = destCount
//        let mutable sourceCount = sourceCount
//
//        let shared = __shared__.Array<'T>(sharedSize) |> __array_to_ptr
//        let sharedIndices = __shared__.Array<int>()
//        let sharedValues = shared
//
//        let tid = threadIdx.x
//        let block = blockIdx.x        
//        let range = ctaLoadBalance destCount indices_global sourceCount block tid countingItr_global mp_global sharedIndices true
//        destCount <- range.y - range.x
//        sourceCount <- range.w - range.z
//
//        let sources = __local__.Array<int>(VT) |> __array_to_ptr
//        deviceSharedToReg (NT * VT) sharedIndices tid sources true
//        deviceMemToMemLoop sourceCount (values_global + range.z) tid sharedValues true
//
//        let values = __local__.Array<'T>(VT) |> __array_to_ptr
//        deviceGather destCount (sharedValues - range.z) sources tid values false
//
//        deviceRegToGlobal destCount values tid (output_global + range.x) false
//    @>
//
//
////type IIntervalExpand<'T> = 
////    {
////        Action : ActionHint -> deviceptr<int> -> deviceptr<'T> -> deviceptr<int> -> deviceptr<int> -> deviceptr<'T> -> unit
////        NumPartitions : int
////    }
//
//
//
////////////////////////////////////////////////////////////////////////////////////
////// IntervalExpand
////
//let intervalExpand (plan:Plan) = cuda {
//    let NT = plan.NT
//    let VT = plan.VT
//    let NV = NT * VT
//
//    let! kernelIntervalExpand = kernelIntervalExpand plan |> Compiler.DefineKernel //"kie"
//    let! mpp = Search.mergePathPartitions MgpuBoundsUpper (comp CompTypeLess 0)
//
//    return Entry(fun program ->
//        let worker = program.Worker
//        let kernelIntervalExpand = program.Apply kernelIntervalExpand
//        //let mpp = mpp.Apply m
//
//        fun (moveCount:int) (intervalCount:int) ->
//            let numBlocks = divup (moveCount + intervalCount) NV
//            let lp = LaunchParam(numBlocks, NT)
//
//            let run (indices_global:deviceptr<int>) (values_global:deviceptr<'T>) (countingItr_global:deviceptr<int>) (mp_global:deviceptr<int>) (output_global:deviceptr<'T>) =
//                fun () ->
//                    
//                    let mpp = mpp moveCount intervalCount NV 0
//                    let partitions = mpp countingItr_global indices_global mp_global                    
//                    kernelIntervalExpand.Launch lp moveCount indices_global values_global intervalCount countingItr_global mp_global output_global
//                |> worker.Eval
//            { NumPartitions = numBlocks + 1 } ) }
//
//
//
////////////////////////////////////////////////////////////////////////////////////
////// KernelIntervalMove
////
//let kernelIntervalMove (plan:Plan) (gather:int) (scatter:int) =
//    let NT = plan.NT
//    let VT = plan.VT
//    let NV = NT * VT
//
//    let gather = if gather = 1 then true else false
//    let scatter = if scatter = 1 then true else false
//
//    let ctaLoadBalance = ctaLoadBalance NT VT
//    let deviceMemToMemLoop = deviceMemToMemLoop NT
//    let deviceGather = deviceGather NT VT
//    let deviceGlobalToReg = deviceGlobalToReg NT VT
//    let deviceScatter = deviceScatter NT VT
//    let deviceRegToGlobal = deviceRegToGlobal NT VT
//    <@ fun  (moveCount          :int) 
//            (gather_global      :deviceptr<int>) 
//            (scatter_global     :deviceptr<int>) 
//            (indices_global     :deviceptr<int>) 
//            (intervalCount      :int) 
//            (input_global       :deviceptr<'T>)
//            (countingItr_global :deviceptr<int>) 
//            (mp_global          :deviceptr<int>) 
//            (output_global      :deviceptr<'T>) 
//            ->
//        let ctaLoadBalance = %ctaLoadBalance
//        let deviceMemToMemLoop = %deviceMemToMemLoop
//        let deviceGather = %deviceGather
//        let deviceGlobalToReg = %deviceGlobalToReg
//        let deviceScatter = %deviceScatter
//        let deviceRegToGlobal = %deviceRegToGlobal
//
//        let mutable moveCount = moveCount
//        let mutable intervalCount = intervalCount
//
//        let indices_shared = __shared__.Array<int>(NT * (VT + 1)) |> __array_to_ptr
//        
//        let tid = threadIdx.x
//        let block = blockIdx.x
//
//        let range = ctaLoadBalance moveCount indices_global intervalCount block tid countingItr_global mp_global indices_shared true
//
//        moveCount <- range.y - range.x
//        intervalCount <- range.w - range.z
//
//        let move_shared = indices_shared
//        let intervals_shared = indices_shared + moveCount
//        let intervals_shared2 = intervals_shared - range.z
//
//        let interval = __local__.Array<int>(VT) |> __array_to_ptr
//        let rank = __local__.Array<int>(VT) |> __array_to_ptr
//
//        for i = 0 to VT - 1 do
//            let index = NT * i + tid
//            let gid = range.x + index
//            interval.[i] <- range.z
//            if index < moveCount then
//                interval.[i] <- move_shared.[index]
//                rank.[i] <- gid - intervals_shared2.[interval.[i]]
//        __syncthreads()
//
//        let gatherArr = __local__.Array<int>(VT) |> __array_to_ptr
//        let scatterArr = __local__.Array<int>(VT) |> __array_to_ptr
//        if gather then
//            deviceMemToMemLoop intervalCount (gather_global + range.z) tid intervals_shared true
//
//            for i = 0 to VT - 1 do
//                gatherArr.[i] <- intervals_shared2.[interval.[i]] + rank.[i]
//            __syncthreads()
//
//        if scatter then
//            deviceMemToMemLoop intervalCount (scatter_global + range.z) tid intervals_shared true
//
//            for i = 0 to VT - 1 do
//                scatterArr.[i] <- intervals_shared2.[interval.[i]] + rank.[i]
//            __syncthreads()
//
//        let data = __local__.Array<'T>(VT) |> __array_to_ptr
//        if gather then
//            deviceGather moveCount input_global gatherArr tid data false
//        else
//            deviceGlobalToReg moveCount (input_global + range.x) tid data false
//
//        if scatter then
//            deviceScatter moveCount data tid scatterArr output_global false
//        else
//            deviceRegToGlobal moveCount data tid (output_global + range.x) false
//    @>
//
//
////type IIntervalGather<'T> =
////    {
////        Action : ActionHint -> deviceptr<int> -> deviceptr<int> -> deviceptr<'T> -> deviceptr<int> -> deviceptr<int> -> deviceptr<'T> -> unit
////        NumPartitions : int
////    }
//
//
////////////////////////////////////////////////////////////////////////////////////
////// IntervalGather
////
//let intervalGather (plan:Plan) = cuda {    
//    let NV = plan.NT * plan.VT
//
//    let! kernelIntervalMove = kernelIntervalMove plan 1 0 |> Compiler.DefineKernel //"kim"
//    let! mpp = Search.mergePathPartitions MgpuBoundsUpper (comp CompTypeLess 0)
//
//    return Entry(fun program ->
//        let worker = program.Worker
//        let kernelIntervalMove = program.Apply kernelIntervalMove
//        //let mpp = mpp.Apply m
//
//        fun (moveCount:int) (intervalCount:int) ->
//            let numBlocks = divup (moveCount + intervalCount) NV
//            let lp = LaunchParam(numBlocks, plan.NT)
//
//            let run (gather_global:deviceptr<int>) (indices_global:deviceptr<int>) (input_global:deviceptr<'T>) (countingItr_global:deviceptr<int>) (mp_global:deviceptr<int>) (output_global:deviceptr<'T>) =
//                fun () ->
//                    
//                    let mpp = mpp moveCount intervalCount NV 0
//                    let partitions = mpp countingItr_global indices_global mp_global
//                    kernelIntervalMove.Launch lp moveCount gather_global (deviceptr(0n)) indices_global intervalCount input_global countingItr_global mp_global output_global
//                |> worker.Eval
//            { NumPartitions = numBlocks + 1 } ) }
//
//
////type IIntervalScatter<'T> = 
////    {
////        Action : ActionHint -> deviceptr<int> -> deviceptr<int> -> deviceptr<'T> -> deviceptr<int> -> deviceptr<int> -> deviceptr<'T> -> unit
////        NumPartitions : int
////    }
//
////////////////////////////////////////////////////////////////////////////////////
////// IntervalScatter
////
//let intervalScatter (plan:Plan) = cuda {    
//    let NV = plan.NT * plan.VT
//    
//    let! kernelIntervalMove = kernelIntervalMove plan 0 1 |> Compiler.DefineKernel //"kim"
//    let! mpp = Search.mergePathPartitions MgpuBoundsUpper (comp CompTypeLess 0)
//    
//    return Entry(fun program ->
//        let worker = program.Worker
//        let kernelIntervalMove = program.Apply kernelIntervalMove
//        //let mpp = mpp.Apply m
//        
//        fun (moveCount:int) (intervalCount:int) ->
//            let numBlocks = divup (moveCount + intervalCount) NV
//            let lp = LaunchParam(numBlocks, plan.NT)
//            
//            let run (scatter_global:deviceptr<int>) (indices_global:deviceptr<int>) (input_global:deviceptr<'T>)  (countingItr_global:deviceptr<int>) (mp_global:deviceptr<int>) (output_global:deviceptr<'T>) =
//                fun () ->
//                    
//                    let mpp = mpp moveCount intervalCount NV 0
//                    let partitions = mpp countingItr_global indices_global mp_global
//                    kernelIntervalMove.Launch lp moveCount (deviceptr<int>(0n)) scatter_global indices_global intervalCount input_global countingItr_global mp_global output_global
//                |> worker.Eval
//            { NumPartitions = numBlocks + 1 } ) }
//
//
////type IIntervalMove<'T> = 
////    {
////        Action : ActionHint -> deviceptr<int> -> deviceptr<int> -> deviceptr<int> -> deviceptr<'T> -> deviceptr<int> -> deviceptr<int> -> deviceptr<'T> -> unit
////        NumPartitions : int
////    }
//
//
////////////////////////////////////////////////////////////////////////////////////
////// IntervalMove
////
//let intervalMove (plan:Plan) = cuda {    
//    let NV = plan.NT * plan.VT
//    
//    let! kernelIntervalMove = kernelIntervalMove plan 1 1 |> Compiler.DefineKernel //"kim"
//    let! mpp = Search.mergePathPartitions MgpuBoundsUpper (comp CompTypeLess 0)
//    
//    return Entry(fun program ->
//        let worker = program.Worker
//        let kernelIntervalMove = program.Apply kernelIntervalMove
//        //let mpp = mpp.Apply m
//        
//        fun (moveCount:int) (intervalCount:int) ->
//            let numBlocks = divup (moveCount + intervalCount) NV
//            let lp = LaunchParam(numBlocks, plan.NT)
//            
//            let run (gather_global:deviceptr<int>) (scatter_global:deviceptr<int>) (indices_global:deviceptr<int>) (input_global:deviceptr<'T>) (countingItr_global:deviceptr<int>) (mp_global:deviceptr<int>) (output_global:deviceptr<'T>) =
//                fun () ->
//                    
//                    let mpp = mpp moveCount intervalCount NV 0
//                    let partitions = mpp countingItr_global indices_global mp_global
//                    kernelIntervalMove.Launch lp moveCount gather_global scatter_global indices_global intervalCount input_global countingItr_global mp_global output_global
//                |> worker.Eval
//            { NumPartitions = numBlocks + 1 } ) }
