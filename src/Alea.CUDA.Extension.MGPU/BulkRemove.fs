﻿module Alea.CUDA.Extension.MGPU.BulkRemove

// this maps to bulkremove.cuh

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
open Alea.CUDA.Extension.MGPU.CTAScan
open Alea.CUDA.Extension.MGPU.Search


// this is plan
type Plan = 
    {
        NT : int
        VT : int
    }

let kernelBulkRemove (plan:Plan) =
    let NT = plan.NT
    let VT = plan.VT
    let NV = NT * VT

    let capacity, scan2 = ctaScan2 NT (scanOp ScanOpTypeAdd 0)
    let alignOfTI, sizeOfTI = TypeUtil.cudaAlignOf typeof<'TI>, sizeof<'TI>
    let alignOfTV, sizeOfTV = TypeUtil.cudaAlignOf typeof<'TV>, sizeof<'TV>
    let sharedAlign = max alignOfTI alignOfTV
    let sharedSize = max (sizeOfTI * NV) (sizeOfTV * capacity)
    let createSharedExpr = createSharedExpr sharedAlign sharedSize
    
    let deviceGlobalToReg = deviceGlobalToReg NT VT
    let deviceSharedToReg = deviceSharedToReg NT VT
    let deviceRegToGlobal = deviceRegToGlobal NT VT
    let deviceGather = deviceGather NT VT
        

    <@ fun (source_global:DevicePtr<'TI>) (sourceCount:int) (indices_global:DevicePtr<int>) (indicesCount:int) (p_global:DevicePtr<int>) (dest_global:DevicePtr<'TV>) ->
        
        let deviceGather = %deviceGather
        
        let deviceGlobalToReg = %deviceGlobalToReg
        let deviceSharedToReg = %deviceSharedToReg
        let deviceRegToGlobal = %deviceRegToGlobal

        let scan = %scan2
//
        let shared = %(createSharedExpr)
        let sharedScan = shared.Reinterpret<int>()
        let sharedIndices = shared.Reinterpret<int>()

        let tid = threadIdx.x
        let block = blockIdx.x
        let gid = block * NV
        
        let mutable sourceCount = sourceCount
        sourceCount <- min NV (sourceCount - gid)

        let mutable source_global = source_global

        // search for begin and end iterators of interval to load
        let p0 = p_global.[block]
        let p1 = p_global.[block + 1]

        // Set the flags to 1. The default is to copy a value
        for i = 0 to VT - 1 do
            let index = NT * i + tid
            if index < sourceCount then sharedIndices.[index] <- 1
        __syncthreads()

        // Load the indices into register
        let begin' = p0
        let indexCount = p1 - begin'
        let indices = __local__<int>(VT).Ptr(0)
        deviceGlobalToReg indexCount (indices_global + begin') tid indices dontSync

        // Set the counter to 0 for each index we've loaded
        for i = 0 to VT - 1 do
            if (NT * i + tid) < indexCount then
                sharedIndices.[indices.[i] - gid] <- 0
        __syncthreads()

        // Run a raking scan over the flags.  We count the set flags - this is the
        // number of elements to load in per thread
        let mutable x = 0
        for i = 0 to VT - 1 do
            indices.[i] <- sharedIndices.[VT * tid + i]
            x <- x + indices.[i]
        __syncthreads()

        // Run a CTA scan and scatter the gather indices to shared memory
        let mutable s = scan tid x sharedScan
        for i = 0 to VT - 1 do
            if indices.[i] = 1 then                
                sharedIndices.[s] <- VT * tid + i
                s <- s + 1                
        __syncthreads()

        // Load the gather indices into register
        deviceSharedToReg NV sharedIndices tid indices doSync
        
        // Gather the data into register.  The number of values to copy
        // is sourceCount - indexCount
        source_global <- source_global + gid
        let count = sourceCount - indexCount
        let values = __local__<'TI>(VT).Ptr(0)
        deviceGather count source_global indices tid values dontSync

        // Store all the valid registers to dest_global
        deviceRegToGlobal count values tid (dest_global.Reinterpret<'TI>() + gid - begin') dontSync  @>

type IBulkRemove<'TI> =
    {
        Action : ActionHint -> DevicePtr<'TI> -> DevicePtr<int> -> DevicePtr<'TI> -> unit        
    }



let bulkRemove (ident:'T) = cuda {
    let plan = { NT = 128; VT = 11 }
    let NV = plan.NT * plan.VT
    
    let! kernelBulkRemove = kernelBulkRemove plan |> defineKernelFunc
    let! bsp = Search.binarySearchPartitions MgpuBoundsLower (comp CompTypeLess ident)

    return PFunc(fun (m:Module) ->
        let worker = m.Worker
        let kernelBulkRemove = kernelBulkRemove.Apply m
        let bsp = bsp.Apply m
        
        fun (sourceCount:int) (indicesCount:int) ->    
            let numBlocks = divup sourceCount NV
            let lp = LaunchParam(numBlocks, plan.NT)
            let bsp = bsp sourceCount indicesCount NV
            let parts = worker.Malloc(numBlocks + 1)
            
            let action (hint:ActionHint) (source_global:DevicePtr<'TI>) (indices_global:DevicePtr<int>) (dest_global:DevicePtr<'TI>) =
                let lp = lp |> hint.ModifyLaunchParam
                let partitions = (bsp.Action hint indices_global parts.Ptr)
                kernelBulkRemove.Launch lp source_global sourceCount indices_global indicesCount parts.Ptr dest_global

            { Action = action } ) }


//        fun (sourceCount:int) (indices_global:DevicePtr<int>) (indicesCount:int) ->
//            let numBlocks = divup sourceCount NV
//            let lp = LaunchParam(numBlocks, plan.NT)
//            let bsp = bsp sourceCount indicesCount NV
//            let parts = worker.Malloc(numBlocks + 1)
//            
//            let action (hint:ActionHint) (source_global:DevicePtr<'TI>) (dest_global:DevicePtr<'TI>) =
//                let lp = lp |> hint.ModifyLaunchParam
//                let partitions = (bsp.Action hint indices_global parts.Ptr)
//                kernelBulkRemove.Launch lp source_global sourceCount indices_global indicesCount parts.Ptr dest_global
//
//            { Action = action } ) }
            
