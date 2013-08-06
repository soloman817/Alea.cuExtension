module Alea.CUDA.Extension.MGPU.IntervalMove

open System.Runtime.InteropServices
open Microsoft.FSharp.Collections
open Alea.CUDA
open Alea.CUDA.Extension
open Alea.CUDA.Extension.Util
open Alea.CUDA.Extension.MGPU
open Alea.CUDA.Extension.MGPU.QuotationUtil
open Alea.CUDA.Extension.MGPU.DeviceUtil
open Alea.CUDA.Extension.MGPU.LoadStore
open Alea.CUDA.Extension.MGPU.CTALoadBalance




//////////////////////////////////////////////////////////////////////////////////
//// KernelIntervalExpand
//
let kernelIntervalExpand (plan:Plan) =
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    let ctaLoadBalance = ctaLoadBalance NT VT
    let deviceSharedToReg = deviceSharedToReg NT VT
    let deviceMemToMemLoop = deviceMemToMemLoop NT
    let deviceGather = deviceGather NT VT
    let deviceRegToGlobal = deviceRegToGlobal NT VT

    let sharedSize = NT * (VT + 1)

    <@ fun  (destCount      :int) 
            (indices_global :DevicePtr<int>) 
            (values_global  :DevicePtr<'T>) 
            (sourceCount    :int)
            (ctaCountingItr  :DevicePtr<int>) 
            (mp_global      :DevicePtr<int>) 
            (output_global  :DevicePtr<'T>) 
            ->
        let ctaLoadBalance = %ctaLoadBalance
        let deviceSharedToReg = %deviceSharedToReg
        let deviceMemToMemLoop = %deviceMemToMemLoop
        let deviceGather = %deviceGather
        let deviceRegToGlobal = %deviceRegToGlobal

        let mutable destCount = destCount
        let mutable sourceCount = sourceCount

        let shared = __shared__<'T>(sharedSize).Ptr(0)
        let sharedIndices = shared.Reinterpret<int>()
        let sharedValues = shared

        let tid = threadIdx.x
        let block = blockIdx.x
        //let countingItr = __local__<int>(sharedSize).Ptr(0)
        let range = ctaLoadBalance destCount indices_global sourceCount block tid ctaCountingItr mp_global sharedIndices true
        destCount <- range.y - range.x
        sourceCount <- range.w - range.z

        let sources = __local__<int>(VT).Ptr(0)
        deviceSharedToReg (NT * VT) sharedIndices tid sources true
        deviceMemToMemLoop sourceCount (values_global + range.z) tid sharedValues true

        let values = __local__<'T>(VT).Ptr(0)
        deviceGather destCount (sharedValues - range.z) sources tid values false

        deviceRegToGlobal destCount values tid (output_global + range.x) false
    @>


type IIntervalExpand<'T> = 
    {
        Action : ActionHint -> DevicePtr<int> -> DevicePtr<'T> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<'T> -> unit
        //Action : ActionHint -> DevicePtr<int> -> DevicePtr<'T> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<'T> -> unit
        NumPartitions : int
    }



//////////////////////////////////////////////////////////////////////////////////
//// IntervalExpand
//
let intervalExpand (plan:Plan) = cuda {
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    let! kernelIntervalExpand = kernelIntervalExpand plan |> defineKernelFuncWithName "kie"
    let! mpp = Search.mergePathPartitions MgpuBoundsUpper (comp CompTypeLess 0)

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelIntervalExpand = kernelIntervalExpand.Apply m
        let mpp = mpp.Apply m

        fun (moveCount:int) (intervalCount:int) ->
            let numBlocks = divup (moveCount + intervalCount) NV
            let lp = LaunchParam(numBlocks, NT)

            let action (hint:ActionHint) (indices_global:DevicePtr<int>) (values_global:DevicePtr<'T>) (ctaCountingItr:DevicePtr<int>) (mpCountingItr:DevicePtr<int>) (mp_global:DevicePtr<int>) (output_global:DevicePtr<'T>) =
                fun () ->
                    let lp = lp |> hint.ModifyLaunchParam
                    let mpp = mpp moveCount intervalCount NV 0
                    let partitions = mpp.Action hint mpCountingItr indices_global mp_global
                    //use ctaCountingItr = worker.Malloc([|0..moveCount|])
                    kernelIntervalExpand.Launch lp moveCount indices_global values_global intervalCount ctaCountingItr mp_global output_global
                |> worker.Eval
            { Action = action; NumPartitions = numBlocks + 1 } ) }



//////////////////////////////////////////////////////////////////////////////////
//// KernelIntervalMove
//
let kernelIntervalMove (plan:Plan) (gather:int) (scatter:int) =
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    let gather = if gather = 1 then true else false
    let scatter = if scatter = 1 then true else false

    let ctaLoadBalance = ctaLoadBalance NT VT
    let deviceMemToMemLoop = deviceMemToMemLoop NT
    let deviceGather = deviceGather NT VT
    let deviceGlobalToReg = deviceGlobalToReg NT VT
    let deviceScatter = deviceScatter NT VT
    let deviceRegToGlobal = deviceRegToGlobal NT VT
    <@ fun  (moveCount      :int) 
            (gather_global  :DevicePtr<int>) 
            (scatter_global :DevicePtr<int>) 
            (indices_global :DevicePtr<int>) 
            (intervalCount  :int) 
            (input_global   :DevicePtr<'T>)
            (ctaCountingItr  :DevicePtr<int>) 
            (mp_global      :DevicePtr<int>) 
            (output_global  :DevicePtr<'T>) 
            ->
        let ctaLoadBalance = %ctaLoadBalance
        let deviceMemToMemLoop = %deviceMemToMemLoop
        let deviceGather = %deviceGather
        let deviceGlobalToReg = %deviceGlobalToReg
        let deviceScatter = %deviceScatter
        let deviceRegToGlobal = %deviceRegToGlobal

        let mutable moveCount = moveCount
        let mutable intervalCount = intervalCount

        let indices_shared = __shared__<int>(NT * (VT + 1)).Ptr(0)
        
        let tid = threadIdx.x
        let block = blockIdx.x

        let range = ctaLoadBalance moveCount indices_global intervalCount block tid ctaCountingItr mp_global indices_shared true

        moveCount <- range.y - range.x
        intervalCount <- range.w - range.z

        let move_shared = indices_shared
        let intervals_shared = indices_shared + moveCount
        let intervals_shared2 = intervals_shared - range.z

        let interval = __local__<int>(VT).Ptr(0)
        let rank = __local__<int>(VT).Ptr(0)

        for i = 0 to VT - 1 do
            let index = NT * i + tid
            let gid = range.x + index
            interval.[i] <- range.z
            if index < moveCount then
                interval.[i] <- move_shared.[index]
                rank.[i] <- gid - intervals_shared2.[interval.[i]]
        __syncthreads()

        let gatherArr = __local__<int>(VT).Ptr(0)
        let scatterArr = __local__<int>(VT).Ptr(0)
        if gather then
            deviceMemToMemLoop intervalCount (gather_global + range.z) tid intervals_shared true

            for i = 0 to VT - 1 do
                gatherArr.[i] <- intervals_shared2.[interval.[i]] + rank.[i]
            __syncthreads()

        if scatter then
            deviceMemToMemLoop intervalCount (scatter_global + range.z) tid intervals_shared true

            for i = 0 to VT - 1 do
                scatterArr.[i] <- intervals_shared2.[interval.[i]] + rank.[i]
            __syncthreads()

        let data = __local__<'T>(VT).Ptr(0)
        if gather then
            deviceGather moveCount input_global gatherArr tid data false
        else
            deviceGlobalToReg moveCount (input_global + range.x) tid data false

        if scatter then
            deviceScatter moveCount data tid scatterArr output_global false
        else
            deviceRegToGlobal moveCount data tid (output_global + range.x) false
    @>


type IIntervalGather<'T> =
    {
        Action : ActionHint -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<'T> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<'T> -> unit
        NumPartitions : int
    }


//////////////////////////////////////////////////////////////////////////////////
//// IntervalGather
//
let intervalGather (plan:Plan) = cuda {    
    let NV = plan.NT * plan.VT

    let! kernelIntervalMove = kernelIntervalMove plan 1 0 |> defineKernelFuncWithName "kim"
    let! mpp = Search.mergePathPartitions MgpuBoundsUpper (comp CompTypeLess 0)

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelIntervalMove = kernelIntervalMove.Apply m
        let mpp = mpp.Apply m

        fun (moveCount:int) (intervalCount:int) ->
            let numBlocks = divup (moveCount + intervalCount) NV
            let lp = LaunchParam(numBlocks, plan.NT)

            let action (hint:ActionHint) (gather_global:DevicePtr<int>) (indices_global:DevicePtr<int>) (input_global:DevicePtr<'T>) (countingItr:DevicePtr<int>) (mp_global:DevicePtr<int>) (output_global:DevicePtr<'T>) =
                fun () ->
                    let lp = lp |> hint.ModifyLaunchParam
                    let mpp = mpp moveCount intervalCount NV 0
                    let partitions = mpp.Action hint countingItr indices_global mp_global
                    kernelIntervalMove.Launch lp moveCount gather_global (DevicePtr(0n)) indices_global intervalCount input_global countingItr mp_global output_global
                |> worker.Eval
            { Action = action; NumPartitions = numBlocks + 1 } ) }


type IIntervalScatter<'T> = 
    {
        Action : ActionHint -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<'T> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<'T> -> unit
        NumPartitions : int
    }

//////////////////////////////////////////////////////////////////////////////////
//// IntervalScatter
//
let intervalScatter (plan:Plan) = cuda {    
    let NV = plan.NT * plan.VT
    
    let! kernelIntervalMove = kernelIntervalMove plan 0 1 |> defineKernelFuncWithName "kim"
    let! mpp = Search.mergePathPartitions MgpuBoundsUpper (comp CompTypeLess 0)
    
    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelIntervalMove = kernelIntervalMove.Apply m
        let mpp = mpp.Apply m
        
        fun (moveCount:int) (intervalCount:int) ->
            let numBlocks = divup (moveCount + intervalCount) NV
            let lp = LaunchParam(numBlocks, plan.NT)
            
            let action (hint:ActionHint) (scatter_global:DevicePtr<int>) (indices_global:DevicePtr<int>) (input_global:DevicePtr<'T>)  (countingItr:DevicePtr<int>) (mp_global:DevicePtr<int>) (output_global:DevicePtr<'T>) =
                fun () ->
                    let lp = lp |> hint.ModifyLaunchParam
                    let mpp = mpp moveCount intervalCount NV 0
                    let partitions = mpp.Action hint countingItr indices_global mp_global
                    kernelIntervalMove.Launch lp moveCount (DevicePtr<int>(0n)) scatter_global indices_global intervalCount input_global countingItr mp_global output_global
                |> worker.Eval
            { Action = action; NumPartitions = numBlocks + 1 } ) }


type IIntervalMove<'T> = 
    {
        Action : ActionHint -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<'T> -> DevicePtr<int> -> DevicePtr<int> -> DevicePtr<'T> -> unit
        NumPartitions : int
    }


//////////////////////////////////////////////////////////////////////////////////
//// IntervalMove
//
let intervalMove (plan:Plan) = cuda {    
    let NV = plan.NT * plan.VT
    
    let! kernelIntervalMove = kernelIntervalMove plan 1 1 |> defineKernelFuncWithName "kim"
    let! mpp = Search.mergePathPartitions MgpuBoundsUpper (comp CompTypeLess 0)
    
    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelIntervalMove = kernelIntervalMove.Apply m
        let mpp = mpp.Apply m
        
        fun (moveCount:int) (intervalCount:int) ->
            let numBlocks = divup (moveCount + intervalCount) NV
            let lp = LaunchParam(numBlocks, plan.NT)
            
            let action (hint:ActionHint) (gather_global:DevicePtr<int>) (scatter_global:DevicePtr<int>) (indices_global:DevicePtr<int>) (input_global:DevicePtr<'T>) (countingItr:DevicePtr<int>) (mp_global:DevicePtr<int>)  (output_global:DevicePtr<'T>) =
                fun () ->
                    let lp = lp |> hint.ModifyLaunchParam
                    let mpp = mpp moveCount intervalCount NV 0
                    let partitions = mpp.Action hint countingItr indices_global mp_global
                    kernelIntervalMove.Launch lp moveCount gather_global scatter_global indices_global intervalCount input_global countingItr mp_global output_global
                |> worker.Eval
            { Action = action; NumPartitions = numBlocks + 1 } ) }
